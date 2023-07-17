# -*- coding: utf-8 -*-
"""
MetamaterialFinder

Automatic (multimaterial) metamaterial finder based on pores.
See: TODO: Add DOI

@author:
    Mathias Fleisch
    Polymer Competence Center Leoben GmbH
    mathias.fleisch@pccl.at
"""

# Python libraries
import os
import numpy as np
import sys
from numpy import array
import itertools
import shutil
import json
import subprocess
import ConfigParser as cp
import inspect
import time

# ABAQUS libraries
from abaqus import *
from abaqusConstants import *
from caeModules import *

# Load additional ABAQUS modules (paths from config.ini)
# Set working directory to file location
src_file_path = inspect.getfile(lambda: None)
# print >> sys.__stdout__, src_file_path
# os.chdir(os.path.dirname(src_file_path))
# Read paths from config.ini
config = cp.ConfigParser()
config.read('config.ini')
path_nearest_node = config.get('ABAQUS', 'path_nearest_node')
path_micromechanics = config.get('ABAQUS', 'path_micromechanics')
# Load modules
# NearestNode
sys.path.append(path_nearest_node)
import nearestNodeModule
# Micromechanics
sys.path.append(path_micromechanics)
import microMechanics
from microMechanics.mmpBackend import Interface
from microMechanics.mmpBackend.mmpInterface.mmpRVEConstants import *
from microMechanics.mmpBackend.mmpKernel.mmpLibrary import mmpRVEGenerationUtils as RVEGenerationUtils
backwardCompatibility.setValues(reportDeprecated=False)

# Load local libraries
import Materials
import PoreFunctions

# Reload module in case new pore functions and/or materials were added
import imp
imp.reload(Materials)
imp.reload(PoreFunctions)

# Global settings
session.journalOptions.setValues(replayGeometry=COORDINATE,
                                 recoverGeometry=COORDINATE)
TOL = 1e-6

n = 10
RELATION_DICT = [{'P{}'.format(i+1): 'comb[{}]'.format(i) for i in range(n)},
                 {'X{}'.format(i+1): '[{}]'.format(i+3) for i in range(n)},
                 {'A{}'.format(i+1): '[{}]'.format(i+2) for i in range(n)},
                 {'LX': 'lx', 'LY': 'ly'}]
RELATION_DICT = {k: v for d in RELATION_DICT for k, v in d.items()}

class MetamaterialFinder(object):
    """
    MetamaterialFinder class
    """
    def __init__(self, name, sheet_size=200):
        """
        Parameters
        ----------
        name : str
            Name of the object.
        sheet_size : int, optional
            Approximate sheet size for the sketch in Abaqus. The default is 200.

        Returns
        -------
        None.

        """
        self.name = name
        self.sheet_size = sheet_size

        # Set up Abaqus model
        Mdb()
        self.model = mdb.models['Model-1']

    def create_materials(self, materials):
        """
        Creates the materials for the simulation.

        Supported material models so far:
            - Linear elastic material

        Parameters
        ----------
        materials : iterable
            List of material objects.

        Returns
        -------
        None.

        """
        self.materials = {}
        for material in materials:
            mat = self.model.Material(name=material.name)
            # Add support for new material models here
            if type(material) == Materials.LinearElasticMaterial:
                mat.Elastic(table=((material.E, material.nu), ))
                mat.Density(table=((material.density, ), ))
            self.materials[material.name] = mat

    def create_sketches(self, pores, side_lengths_uc, N=None,
                        side_lengths_full=None, force_full=False,
                        force_homogenization=False):
        """
        Creates the sketches for the outside and pores.

        Parameters
        ----------
        pores : dict
            Dictionary of pores with coordinates and material.
        side_lengths_uc : iterable
            Side lenghts of the unit cell.
        N : iterable or None, optional
            Amount of unit cells in x- and y-direction. The default is None.
        side_lengths_full : iterable or None, optional
            Side lengths of outer region. The default is None.
        force_full : bool, optional
            Force a full-scale simulation even if it is possible to perform
            a homogenization. The default is False.
        force_homogenization : bool, optional
            Force a homogenization simulation, even if the pores extend the
            unit cell. The default is False.

        Returns
        -------
        None.

        """
        self.pores = pores
        # Check if pores extend the unit cell
        # If not, a homogenization simulation can be used
        LX, LY = side_lengths_uc
        points = [p['coordinates'] for p in pores]
        inside = [True if max(op[0])<LX and min(op[0])>0 and max(op[1])<LY and min(op[1])>0 else False for op in points]
        if not force_full and all(inside):
            self._sketch_homogenization(points, side_lengths_uc)
        else:
            self._sketch_full_simulation(points, side_lengths_uc, N,
                                         side_lengths_full)
        self.LX, self.LY = LX, LY
        if force_homogenization:
            self.homogenization = True

    def _sketch_homogenization(self, points, side_lengths_uc):
        """
        Creates the sketch for a homogenization simulation. The following
        sketches are created:
            - A rectangle with side lenghts LX x LY in the first quadrant, starting at (0, 0)
            - The pores

        Parameters
        ----------
        points : iterable
            List of numpy.ndarrays with pore coordinates
        side_lengths_uc : iterable
            LX x LY coordinate pair

        Returns
        -------
        None.

        """
        self.homogenization = True
        LX, LY = side_lengths_uc
        # Sketch rectangle of which the pores are cut out
        sketch = self.model.ConstrainedSketch(name='{}-cut'.format(self.name),
                                              sheetSize=self.sheet_size)
        sketch.Line(point1=(0,0), point2=(LX, 0))
        sketch.Line(point1=(LX, 0), point2=(LX, LY))
        sketch.Line(point1=(LX, LY), point2=(0, LY))
        sketch.Line(point1=(0, LY), point2=(0, 0))
        self.sketch_cut = sketch
        # Sketch pores
        for i, p in enumerate(self.pores):
            sketch = self.model.ConstrainedSketch(name='{}-pore-{}'.format(self.name, i),
                                                  sheetSize=self.sheet_size)
            x, y = p['coordinates'][0], p['coordinates'][1]
            spline_point_pairs = zip(x, y)
            sketch.Spline(points=spline_point_pairs)
            p['sketch'] = sketch
            p['start-points'] = [(spline_point_pairs[0][0], spline_point_pairs[0][1])]
            # Generate bounding box
            p['bounding-boxes'] = [((min(x), min(y)), (max(x), (max(y))))]
        self.middle_point_top = (LX/2., LY, 0)
        self.width = LX
        self.height = LY

    def _sketch_full_simulation(self, points, side_lengths_uc, N,
                                side_lengths_full):
        """
        Creates the sketch for a full-scale simulation with NX x NY unit cells.

        Parameters
        ----------
        points : iterable
            List of numpy.ndarrays with pore coordinates
        side_lengths_uc : iterable
            LX x LY coordinate pair
        N : iterable
            NX x NY pair of integers.
        side_lengths_full : iterable
            Pair of side lengths of the outer region

        Returns
        -------
        None.

        """
        self.homogenization = False
        # Sketch recangle of which the pores are cut out
        # Since the first unit cell boundary box starts at (0, 0) in the first
        # quadrant, we need to recalculate the starting points of the rectangle
        # to consider the outer dimensions of the full structure
        LX, LY = side_lengths_uc
        NX, NY = N
        AX, AY = side_lengths_full
        start_x = -(AX-NX*LX)/2.
        start_y = -(AY-NY*LY)/2.
        p1 = (start_x, start_y)
        p2 = (start_x+AX, start_y)
        p3 = (start_x+AX, start_y+AY)
        p4 = (start_x, start_y+AY)
        self.bottom = start_y
        self.top = start_y+AY
        self.left = start_x
        self.right = start_x+AX
        self.middle_point_top = ((self.right+self.left)/2., self.top, 0)
        middle_height = (self.bottom+self.top)/2.
        self.middle_point_left = (self.left, middle_height, 0)
        self.middle_point_right = (self.right, middle_height, 0)
        self.width = AX
        self.height = AY
        sketch = self.model.ConstrainedSketch(name='{}-cut'.format(self.name),
                                              sheetSize=self.sheet_size)
        sketch.Line(point1=p1, point2=p2)
        sketch.Line(point1=p2, point2=p3)
        sketch.Line(point1=p3, point2=p4)
        sketch.Line(point1=p4, point2=p1)
        self.sketch_cut = sketch
        # Sketch pores
        for i, p in enumerate(self.pores):
            start_points = []
            bounding_boxes = []
            sketch = self.model.ConstrainedSketch(name='{}-pore-{}'.format(self.name, i),
                                                  sheetSize=self.sheet_size)
            for i in range(NX):
                for j in range(NY):
                    x, y = p['coordinates'][0], p['coordinates'][1]
                    sp_x = x+i*LX
                    sp_y = y+j*LY
                    spline_points = zip(sp_x, sp_y)
                    sketch.Spline(points=spline_points)
                    start_points.append((spline_points[0][0], spline_points[0][1]))
                    bounding_boxes.append(((min(sp_x), min(sp_y)), (max(sp_x), max(sp_y))))
            p['sketch'] = sketch
            p['start-points'] = start_points
            p['bounding-boxes'] = bounding_boxes

    def create_parts(self, depth=1, dimension='2D'):
        """
        Creates the parts, based on the sketches

        Parameters
        ----------
        depth : float, optional
            Depth of the extrusion. The default is 1.
        dimension : str, optional
            '2D' or '3D'. The default is '2D'.

        Returns
        -------
        None.

        """
        if dimension=='3D' or self.homogenization:
            # Extrude outer region
            p = self.model.Part(name='{}-cut'.format(self.name), dimensionality=THREE_D,
                                type=DEFORMABLE_BODY)
            p.BaseSolidExtrude(sketch=self.sketch_cut, depth=depth)
            # Cut out empty pores
            self._create_cutouts(p, depth, '3D')
            part = mdb.models['Model-1'].parts['{}-part'.format(self.name)]
            # Create filled pores as partitions
            # Extrusion direction
            axis = part.DatumAxisByPrincipalAxis(principalAxis=ZAXIS)
            for pore in self.pores:
                # Find better way to find face
                face = part.faces.findAt(((0.1, 0.1, 0), ))
                edge = part.edges.findAt(coordinates=self.middle_point_top)
                sketch = pore['sketch']
                material = pore['material']
                if material != 'empty':
                    # Create sketch partition
                    part.PartitionFaceBySketch(sketchUpEdge=edge, faces=face,
                                               sketchOrientation=TOP, sketch=sketch)
                    partitions = []
                    # Extrude sketched partition
                    for start_point in pore['start-points']:
                        cells = part.cells[:]
                        edge_pore = (part.edges.findAt(coordinates=(start_point[0], start_point[1], 0)), )
                        partition = part.PartitionCellByExtrudeEdge(line=part.datums[axis.id],
                                                                    cells=cells,
                                                                    edges=edge_pore,
                                                                    sense=FORWARD)
                        partitions.append(partition)
                    pore['partitions'] = partitions
                else:
                    pore['partitions'] = []
            self.dimension = '3D'
            self.depth = depth
        elif dimension=='2D':
            # Planar element for outer region
            p = self.model.Part(name='{}-cut'.format(self.name), dimensionality=TWO_D_PLANAR,
                                type=DEFORMABLE_BODY)
            p.BaseShell(sketch=self.sketch_cut)
            # Cut out empty pores
            self._create_cutouts(p, None, '2D')
            part = mdb.models['Model-1'].parts['{}-part'.format(self.name)]
            # Create filled pores as partitions
            for pore in self.pores:
                # Find better way to find face
                face = part.faces.findAt(((0.1, 0.1, 0), ))
                edge = part.edges.findAt(coordinates=self.middle_point_top)
                sketch = pore['sketch']
                material = pore['material']
                if material != 'empty':
                    # Create sketch partition
                    partition = part.PartitionFaceBySketch(sketchUpEdge=edge, faces=face,
                                                           sketchOrientation=TOP, sketch=sketch)
                    pore['partitions'] = [partition]
                else:
                    pore['partitions'] = []
            self.dimension = '2D'
            self.depth = depth
        self.part = part

    def _create_cutouts(self, part_cut, depth, dimension):
        """
        Cuts out the empty pores. Cutting is only possible in an assembly. Thus
        a temporary assembly with the pores is created.

        Parameters
        ----------
        part_cut : Abaqus part object
            Part to be cutted out.
        depth : float
            Depth of the extrusion.
        dimension : str
            Dimension of the analysis '2D' or '3D'.

        Returns
        -------
        None.

        """
        for i, pore in enumerate(self.pores):
            sketch = pore['sketch']
            if dimension=='2D':
                p = self.model.Part(name='{}-pore-{}'.format(self.name, i), dimensionality=TWO_D_PLANAR,
                                    type=DEFORMABLE_BODY)
                p.BaseShell(sketch=sketch)
            elif dimension=='3D':
                p = self.model.Part(name='{}-pore-{}'.format(self.name, i), dimensionality=THREE_D,
                                    type=DEFORMABLE_BODY)
                p.BaseSolidExtrude(sketch=sketch, depth=depth)
            pore['part'] = p
        # Create temporary assembly to cut out empty pores
        a = mdb.models['Model-1'].rootAssembly
        for i, pore in enumerate(self.pores):
            inst = a.Instance(name='pore-{}'.format(i), part=pore['part'], dependent=ON)
            pore['instance'] = inst
        a.Instance(name='cut', part=part_cut, dependent=ON)
        # Cut assembly
        try:
            a.InstanceFromBooleanCut(name='{}-part'.format(self.name), 
                                     instanceToBeCut=mdb.models['Model-1'].rootAssembly.instances['cut'], 
                                     cuttingInstances=[pore['instance'] for pore in self.pores if pore['material']=='empty'], 
                                     originalInstances=SUPPRESS)
        except:
            # Create copy if no pores were to cut out
            mdb.models['Model-1'].Part(name='{}-part'.format(self.name),
                                       objectToCopy=part_cut)

    def create_mesh(self, mesh_size, deviation_factor, min_size_factor,
                    element_shape, element_type, mesh_size_z=None):
        """
        Creates the mesh.

        Details: https://classes.engineering.wustl.edu/2009/spring/mase5513/abaqus/docs/v6.6/books/usi/default.htm?startat=pt03ch17s04s04.html

        Parameters
        ----------
        mesh_size : float
            Approximate global size.
        deviation_factor : float
            Maximum deviation factor (for circular elements), 0.0 < h/L < 1.0.
        min_size_factor : float
            Minimum size control (by fraction of global size), 0.0 < min < 1.0.
        element_shape : str
            Shape of the elements to be used in the study.
        element_type : str, optional
            Type of the elements to be used in the study.

        Note: If the specified element shape does not match the available shapes
              for the specified dimension, standard values are used
              3D -> hex, quadratic
              2D -> quad, quadratic-plane-stress

        Returns
        -------
        None.

        """
        elements = {
            # 2D
            'quad': {'shape': QUAD,
                     'linear-plane-stress': [CPS4R, CPS3],
                     'linear-plane-strain': [CPE4R, CPE3],
                     'quadratic-plane-stress': [CPS8R, CPS6M],
                     'quadratic-plane-strain': [CPE8R, CPE6M]},
            'quad-dominated': {'shape': QUAD_DOMINATED,
                               'linear': [CPS4R, CPS3],
                               'quadratic': [CPS8R, CPS6M]},
            'tri': {'shape': TRI,
                    'linear': [CPS4R, CPS3],
                    'quadratic': [CPS8R, CPS6M]},
            # 3D
            'hex': {'shape': HEX,
                    'linear': [C3D8R, C3D6, C3D4],
                    'quadratic': [C3D20R, C3D15, C3D10]},
            'hex-dominated': {'shape': HEX_DOMINATED,
                              'linear': [C3D8R, C3D6, C3D4],
                              'quadratic': [C3D20R, C3D15, C3D10]},
            'tet': {'shape': TET,
                    'linear': [C3D8R, C3D6, C3D4],
                    'quadratic': [C3D20R, C3D15, C3D10]},
            'wedge': {'shape': WEDGE,
                      'linear': [C3D8R, C3D6, C3D4],
                      'quadratic': [C3D20R, C3D15, C3D10]}}

        elements_3d = ['hex', 'hex-dominated', 'tet', 'wedge']
        elements_2d = ['quad', 'quad-dominated', 'tri']
        p = self.part
        if self.dimension=='3D':
            region = p.cells[:]
            elems = elements_3d
        elif self.dimension=='2D':
            region = p.faces[:]
            elems = elements_2d
        # Set element shape
        # Make sure the shape matches the dimension, otherwise fall back
        # to default
        if element_shape not in elems:
            if self.dimension == '3D':
                element_shape = 'hex'
                element_type = 'quadratic'
            elif self.dimension == '2D':
                element_shape = 'quad'
                element_type = 'quadratic-plane-stress'
        # Create seeds in z-direction (can be rougher for 2.5D simulations)
        if (self.dimension=='3D') and not self.homogenization:
            e = p.edges
            edge = e.findAt(((self.left, self.bottom, self.depth/2.), ))
            p.seedEdgeBySize(edges=edge, size=mesh_size_z,
                             deviationFactor=deviation_factor, constraint=FINER)
        p.setMeshControls(regions=region, elemShape=elements[element_shape]['shape'])
        element_types = [mesh.ElemType(elemCode=ec) for ec in elements[element_shape][element_type]]
        p.setElementType(regions=(region, ), elemTypes=element_types)
        p.seedPart(size=mesh_size, deviationFactor=deviation_factor,
                    minSizeFactor=min_size_factor)
        p.generateMesh()
        # Check mesh
        mesh_qual = p.verifyMeshQuality(criterion=ANALYSIS_CHECKS)
        failed_elem = mesh_qual['failedElements']

    def create_sets(self):
        """
        Creates the sets.

        Returns
        -------
        None.

        """
        p = self.part
        # Reference point, top and bottom surfaces (or edges)
        # Only needed if a full-scale simulation is used
        if not self.homogenization:
            # Reference point
            rp = p.ReferencePoint(point=self.middle_point_top)
            p.Set(name='RP', referencePoints=(p.referencePoints[rp.id],))
            # Top and bottom surfaces (or edges)
            # Top and bottom surfaces
            if self.dimension=='3D':
                bottom_faces = p.faces.getByBoundingBox(self.left-0.1, self.bottom, -self.depth,
                                                        self.right+0.1, self.bottom, self.depth)
                p.Set(faces=bottom_faces, name='bottom')
                top_faces = p.faces.getByBoundingBox(self.left-0.1, self.top, -self.depth,
                                                     self.right+0.1, self.top, self.depth)
                p.Set(faces=top_faces, name='top')
            elif self.dimension=='2D':
                bottom_edges = p.edges.getByBoundingBox(self.left-0.1, self.bottom, -0.1,
                                                        self.right+0.1, self.bottom, 0.1)
                p.Set(edges=bottom_edges, name='bottom')
                top_edges = p.edges.getByBoundingBox(self.left-0.1, self.top, -0.1,
                                               self.right+0.1, self.top, 0.1)
                p.Set(edges=top_edges, name='top')
        # Sets to apply materials
        # Pores
        for i, pore in enumerate(self.pores):
            if pore['material'] != 'empty':
                set_names = []
                for j, bb in enumerate(pore['bounding-boxes']):
                    x_min, y_min = bb[0]
                    x_max, y_max = bb[1]
                    name = 'pore-{}-{}'.format(i, j)
                    if self.dimension=='3D':
                        c = p.cells.getByBoundingBox(x_min-0.1, y_min-0.1, -0.1, x_max+0.1, y_max+0.1, self.depth+0.1)
                        p.Set(cells=c, name=name)
                    elif self.dimension=='2D':
                        f = p.faces.getByBoundingBox(x_min-0.1, y_min-0.1, -0.1, x_max+0.1, y_max+0.1, 0.1)
                        p.Set(faces=f, name=name)
                    set_names.append(name)
                pore['set-name'] = set_names
            else:
                pore['set-name'] = [None]
        # Outside
        pore_materials = list(set([x['material'] for x in self.pores]))
        try:
            pore_materials.remove('empty')
        except:
            pass
        if (self.dimension=='3D') and (len(pore_materials)==0):
            cell = p.cells[:]
            p.Set(cells=cell, name='outside')
        elif self.dimension=='3D':
            cell = p.cells.findAt(((0.1, 0.1, 0), ))
            p.Set(cells=cell, name='outside')
        elif (self.dimension=='2D') and (len(pore_materials)==0):
            face = p.faces[:]
            p.Set(faces=face, name='outside')
        elif self.dimension=='2D':
            face = p.faces.findAt(((0.1, 0.1, 0), ))
            p.Set(faces=face, name='outside')

    def create_sections(self):
        """
        Creates the sections.
    
        """
        for name in self.materials:
            self.model.HomogeneousSolidSection(name='{}'.format(name),
                                               material=name,
                                               thickness=self.depth)

    def assign_sections(self, outside_material_name):
        """
        Assign sections to outside and pores.

        Parameters
        ----------
        outside_material_name : str
            Name of the material used for the surrounding.

        Returns
        -------
        None.

        """
        p = self.part
        # Outside
        p.SectionAssignment(region=p.sets['outside'],
                            sectionName=outside_material_name,
                            thicknessAssignment=FROM_SECTION)
        # Pores
        for pore in self.pores:
            material = str(pore['material'])
            if material != 'empty':
                set_names = pore['set-name']
                for set_name in set_names:
                    p.SectionAssignment(region=p.sets[set_name],
                                        sectionName=material,
                                        thicknessAssignment=FROM_SECTION)

    def create_assembly(self, extensometer):
        """
        Creates an assembly from the part and adds nodes for boundary
        conditions and post-processing

        Returns
        -------
        None.

        """
        a = self.model.rootAssembly
        # Remove previously created assembly instances to avoid interference
        for i, _ in enumerate(self.pores):
            del a.features['pore-{}'.format(i)]
        del a.features['cut']
        try:
            del a.features['{}-part-1'.format(self.name)]
        except KeyError:
            pass
        inst = a.Instance(name='asm', part=self.part, dependent=ON)
        # Nodes for extensometers
        if not self.homogenization:
            # Align view (needed for plugin)
            session.viewports['Viewport: 1'].setValues(displayedObject=a)
            # Create extensometer nodes
            nodes = a.instances['asm'].nodes
            self.extensometer_ids = {}
            for name, ext_points in extensometer.items():
                x, y = ext_points
                z = 0
                node = nearestNodeModule.findNearestNode(xcoord=x, ycoord=y, zcoord=z,
                                                         name='', instanceName='asm')
                id_node = node[0]
                a.Set(nodes=nodes[id_node-1:id_node], name=str(name))
                self.extensometer_ids[name] = id_node
            # Bottom left node to prevent rigid body motion
            x = self.left
            y = self.bottom
            z = 0.0
            nn_bottom_left = nearestNodeModule.findNearestNode(xcoord=x, ycoord=y, zcoord=z,
                                                               name='', instanceName='asm')
            id_bottom_left = nn_bottom_left[0]
            a.Set(nodes=nodes[id_bottom_left-1:id_bottom_left], name='bottom-left')
            a.regenerate()
        self.instance = a

    def create_constraints(self):
        """
        Couples the reference point to the top surface (only needed for full
                                                        scale simulations)

        Returns
        -------
        None.

        """
        if not self.homogenization:
            # Couple movement of RP to top surface
            self.model.Equation(name='constraint-x', terms=((1.0, 'asm.top', 1), (-1.0, 'asm.RP', 1)))
            self.model.Equation(name='constraint-y', terms=((1.0, 'asm.top', 2), (-1.0, 'asm.RP', 2)))

    def create_steps_and_history(self, nlgeom=False, stepsize=0.05):
        """
        Creates the steps and history if a full-scale simulation is used.

        Returns
        -------
        None.

        """
        if not self.homogenization:
            # Step
            if nlgeom:
                self.model.StaticStep(name='Step-1', previous='Initial', nlgeom=ON)
            else:
                self.model.StaticStep(name='Step-1', previous='Initial', nlgeom=OFF)
            self.model.steps['Step-1'].setValues(initialInc=stepsize, minInc=1e-09, maxInc=stepsize, maxNumInc=1000)
            # History
            self.model.HistoryOutputRequest(name='H-Output-1', createStepName='Step-1',
                                            variables=('U2', 'RF2'),
                                            region=self.instance.sets['asm.RP'])
            for i, name in enumerate(self.extensometer_ids):
                if self.dimension=='3D':
                    self.model.HistoryOutputRequest(name='H-Output-{}'.format(i+2), createStepName='Step-1',
                                                    variables=('U1', 'U2', 'COOR1', 'COOR2', 'COOR3'),
                                                    region=self.instance.sets[str(name)])
                else:
                    self.model.HistoryOutputRequest(name='H-Output-{}'.format(i+2), createStepName='Step-1',
                                                    variables=('U1', 'U2', 'COOR1', 'COOR2'),
                                                    region=self.instance.sets[str(name)])
            self.nlgeom = nlgeom

    def create_amplitude(self):
        """
        Create an amplitude if a full-scale simulation is used.

        Returns
        -------
        None.

        """
        if not self.homogenization:
            self.model.TabularAmplitude(name='amplitude', timeSpan=STEP, 
                                        smooth=SOLVER_DEFAULT, data=((0.0, 0.0), (1.0, 1.0)))

    def save_model(self, temp_path):
        """
        Save the model as an odb-database

        Returns
        -------
        None.

        """
        self.path_odb = os.path.join(temp_path, '{}.odb'.format(self.name))
        mdb.saveAs(self.path_odb)

    def create_boundaries(self, uy=None):
        """
        Creates the boundary conditions, bottom fixed, load on top if full-scale
        simulations are used

        Parameters
        ----------
        uy : float or None, optional
            Displacement in y-direction [in % of total length].
            > 0: Tensile
            < 0: Compression
            The default is None.

        Returns
        -------
        None.

        """
        if not self.homogenization:
            # Fix bottom
            self.model.DisplacementBC(name='fixed-y', createStepName='Step-1', 
                                      region=self.instance.sets['asm.bottom'],
                                      u1=SET, u2=SET, ur3=SET, amplitude=UNSET, 
                                      distributionType=UNIFORM, fieldName='', localCsys=None)
            # self.model.DisplacementBC(name='ridid-body', createStepName='Step-1', 
            #                           region=self.instance.sets['bottom-left'],
            #                           u1=SET, u2=UNSET, ur3=UNSET, amplitude=UNSET, 
            #                           distributionType=UNIFORM, fieldName='', localCsys=None)
            # Move RP in y
            disp_x = ((self.top - self.bottom)/100)*uy[0]
            disp_y = ((self.top - self.bottom)/100.)*uy[1]
            self.model.DisplacementBC(name='load', createStepName='Step-1',
                                      region=self.instance.sets['asm.RP'],
                                      u1=disp_x, u2=disp_y, u3=0, ur1=0, ur2=0, ur3=0, 
                                      amplitude='amplitude', fixed=OFF,
                                      distributionType=UNIFORM, fieldName='', 
                                      localCsys=None)
            self.model.rootAssembly.regenerate()
            # self.uy = uy

    def run_model(self):
        """
        Runs the simulation for the model

        Returns
        -------
        None.

        """
        job = mdb.Job(name=self.name, model='Model-1', type=ANALYSIS,
                      resultsFormat=ODB)
        if self.homogenization:
            Interface.Loading.MechanicalModelMaker(
                    constraintType='PERIODIC', 
                    drivenField='STRAIN', modelName='Model-1', jobName=self.name, 
                    doNotSubmit=False, homogenizeProperties=(True, False, False))
        else:
            job.submit(consistencyChecking=OFF)
        job.waitForCompletion()

    def post_process(self, tmp_path, save_path, param):
        if self.homogenization:
            Interface.PostProcess.MechanicalPostProcessWorkflow(
                    model='Model-1', ODBName=self.path_odb,
                    doHomogenization=True, materialType=('Orthotropic', ),
                    fieldAveraging=False, getStrainConcentration=False, 
                    averageVolume=False, averageVolumeBySection=False,
                    selectedFields=(), getStatisticalDistribution=False)
            src = os.path.join(tmp_path, '{}-StiffnessMatrix-0pt0.txt'.format(self.name))
            with open(src, 'a') as f:
                f.write('Parameters: {}'.format(str(param)))
            shutil.copy(src, save_path)
        else:
            odb = session.openOdb(self.path_odb)
            # vp = session.viewports['Viewport: 1']
            # vp.setValues(displayedObject=odb)
            # # Save image
            # session.printOptions.setValues(vpBackground=ON)
            # session.viewports['Viewport: 1'].view.rotate(xAngle=0, yAngle=0, zAngle=0, 
            #                                              mode=TOTAL)
            # session.viewports['Viewport: 1'].odbDisplay.display.setValues(
            #         plotState=(CONTOURS_ON_DEF,))
            # session.viewports['Viewport: 1'].odbDisplay.contourOptions.setValues(
            #         showMaxLocation=ON)
            # path_png = os.path.join(save_path, '{}-mises.png'.format(self.name))
            # session.printToFile(fileName=path_png, format=PNG,
            #                     canvasObjects=(session.viewports['Viewport: 1'], ))
            step = odb.steps['Step-1']
            hr = [i for i in step.historyRegions.values() if i.name.startswith('Node ')]
            names = []
            data = []
            for node in hr:
                for name, node_id in self.extensometer_ids.items():
                    if node.name == 'Node ASM.{}'.format(node_id):
                        coor1 = np.array(node.historyOutputs['COOR1'].data)
                        coor2 = np.array(node.historyOutputs['COOR2'].data)
                        if not self.nlgeom:
                            u1 = np.array(node.historyOutputs['U1'].data)
                            u2 = np.array(node.historyOutputs['U2'].data)
                            coor1 += u1
                            coor2 += u2
                        names.append('{}-x'.format(name))
                        names.append('{}-y'.format(name))
                        data.append(np.column_stack((coor1[:,1], coor2[:,1])))
                        break
                else:
                    res_u = np.array(node.historyOutputs['U2'].data)
                    res_rf = np.array(node.historyOutputs['RF2'].data)  
            res = np.column_stack([res_u[:,1], res_rf[:,1]]+data)
            # Header with information on initial lengths
            header = 'Length: {}\nWidth: {}\nThickness: {}\n'.format(self.height, self.width, self.depth)
            header += 'Parameters: {}\n'.format(str(param))
            header += 'Displacement Force '
            header += ' '.join(names)
            header += '\n'
            # Write result to file
            path_dat = os.path.join(save_path, '{}-results.dat'.format(self.name))
            with open(path_dat, 'w') as f:
                f.write(header)
                np.savetxt(f, res)
        # Save image of undeformed geometry with material sections
        odb = session.openOdb(self.path_odb)
        vp = session.viewports['Viewport: 1']
        vp.setValues(displayedObject=odb)
        session.printOptions.setValues(vpBackground=ON)
        session.viewports['Viewport: 1'].view.rotate(xAngle=0, yAngle=0, zAngle=0, 
                                                     mode=TOTAL)
        cmap=session.viewports['Viewport: 1'].colorMappings['Material']
        session.viewports['Viewport: 1'].setColor(colorMapping=cmap)
        path_png = os.path.join(save_path, '{}-sections.png'.format(self.name))
        session.viewports['Viewport: 1'].viewportAnnotationOptions.setValues(state=OFF)
        session.viewports['Viewport: 1'].viewportAnnotationOptions.setValues(title=OFF)
        session.printToFile(fileName=path_png, format=PNG,
                            canvasObjects=(session.viewports['Viewport: 1'], ))


def prepare_simulations(simulation_database, pore_database):
    """
    Reads the json database (Simulations.json) and creates all permuations
    of specified parameters

    Parameters
    ----------
    simulation_database : list
        List of database entries

    Returns
    -------
    None.

    """
    simulation_data = {}
    for name, simulation in simulation_database.iteritems():
        active = bool(simulation['Active'])
        if not active:
            # Skip entry if not active
            continue
        lx, ly, lz = [float(x) for x in simulation['LX|LY|LZ'].split('|')]
        nx, ny = [int(x) for x in simulation['NX|NY'].split('|')]
        sx, sy = [float(x) for x in simulation['SX|SY'].split('|')]
        outside_material = str(simulation['Outside material'])
        mesh_name = str(simulation['Mesh'])
        # Create all permutations of parameters
        pore_name = simulation['Pores']
        p = pore_database[pore_name]
        permutation_list = []
        for pf, pfp, pfd, pa in zip(*[p['Pore functions'], p['Pore function parameters'],
                                      p['Pore displacements'], p['Pore angles']]):
            # Create numpy arrays for each range of parameters
            pore_parameter_list = []
            for param_set in pfp.split('|'):
                start, stop, n = [float(x) for x in param_set.split(';')]
                if n == 0:
                    n = 1
                else:
                    n = int(n)
                pore_parameters = np.linspace(start, stop, n)
                pore_parameter_list.append(pore_parameters)
            start, stop, n = [float(x) for x in pa.split(';')]
            if n == 0:
                n = 1
            else:
                n = int(n)
            pore_angles = np.linspace(start, stop, n)
            start, stop, n = [eval(x.replace('LX', str(lx)).replace('LY', str(ly))) for x in pfd['X'].split(';')]
            if n == 0:
                n = 1
            else:
                n = int(n)
            pore_displacements_x = np.linspace(start, stop, n)
            start, stop, n = [eval(x.replace('LX', str(lx)).replace('LY', str(ly))) for x in pfd['Y'].split(';')]
            if n == 0:
                n = 1
            else:
                n = int(n)
            pore_displacements_y = np.linspace(start, stop, n)
            # Create all permutations
            permutations = list(itertools.product(*[pore_displacements_x, pore_displacements_y, pore_angles]+pore_parameter_list))
            permutation_list.append(permutations)
        combinations = list(itertools.product(*permutation_list))
        print('{} combinations found'.format(len(combinations)))
        try:
            # Filter all combinations by the given relations
            relations = p['Pore relations']
            filtered_combinations = []
            print('Filter combinations by given relations (this might take a while) ...')
            for comb in combinations:
                relations_fullfiled = [False for _ in range(len(relations))]
                for i, relation in enumerate(relations):
                    for word, initial in RELATION_DICT.items():
                        relation = relation.replace(word, initial)
                    if eval(relation):
                        relations_fullfiled[i] = True
                if all(relations_fullfiled):
                    filtered_combinations.append(comb)
            combinations = filtered_combinations
        except:
            pass
        pf_param_list = []
        for c in combinations:
            pf_param = [(pf, x, mat) for (pf, x), mat in zip(zip(p['Pore functions'], c), p['Pore materials'])]
            pf_param_list.append(pf_param)
        # Copy pores if multiple displacements/angles are defined, but only a single pore
        combinations_copied = []
        if (len(p['Pore functions'])==1) and (len(p['Pore displacements'])>1):
            if len(p['Pore displacements'])==len(p['Pore angles']):
                print('Single pore with multiple displacements and angles found, copying ...')
                for pf_param in pf_param_list:
                    c = [pf_param[0]]
                    for disp, angle in zip(p['Pore displacements'][1:], p['Pore angles'][1:]):
                        a = float(angle.split(';')[0])
                        xo, _, _ = [eval(x.replace('LX', str(lx)).replace('LY', str(ly))) for x in disp['X'].split(';')]
                        yo, _, _ = [eval(x.replace('LX', str(lx)).replace('LY', str(ly))) for x in disp['Y'].split(';')]
                        new_values = (xo, yo, a) + tuple((p for p in pf_param[0][1][3:]))
                        new_entry = (pf_param[0][0], new_values, pf_param[0][-1])
                        c.append(new_entry)
                    combinations_copied.append(c)
                pf_param_list = combinations_copied
            else:
                print('Single pore with multiple displacements found, but not mutliple angles, skipping copying')
        force_full = bool(simulation['Force full'])
        force_homogenization = bool(simulation['Force homogenization'])
        uy = [float(x) for x in simulation['DX|DY'].split('|')]
        nlgeom = bool(simulation['Nlgeom'])
        stepsize = float(simulation['Stepsize'])
        dimension = str(simulation['Dimension'])
        try:
            extensometer = simulation['Extensometer']
            ext = {}
            for ext_name, coor_list in extensometer.items():
                ext[ext_name] = [eval(x.replace('LX', str(lx)).replace('LY', str(ly))) for x in coor_list]
        except KeyError:
            ext = None
        save_dir = str(simulation['Save directory'])
        simulation_data[name] = {'LX': lx, 'LY': ly, 'LZ': lz, 'NX': nx, 'NY': ny,
                                 'SX': sx, 'SY': sy, 'Outside material': outside_material,
                                 'Mesh name': mesh_name, 'Parameters': pf_param_list,
                                 'Force full': force_full,
                                 'Force homogenization': force_homogenization,
                                 'UY': uy, 'Nlgeom': nlgeom, 'Stepsize': stepsize,
                                 'Dimension': dimension, 'Extensometer': ext,
                                 'Save directory': save_dir}
        return simulation_data


if __name__ == '__main__':

    ###################
    # Load json files #
    ###################

    # Get database paths
    materials_json_path = config.get('DATABASES', 'materials_json_path')
    meshes_json_path = config.get('DATABASES', 'meshes_json_path')
    pores_json_path = config.get('DATABASES', 'pores_json_path')
    simulations_json_path = config.get('DATABASES', 'simulations_json_path')

    # Materials 
    with open(materials_json_path, 'r') as f:
        material_db = json.load(f)
    # Create material objects
    materials = []
    for class_name, object_list in material_db.items():
        for object_dict in object_list:
            mat = getattr(Materials, class_name)
            materials.append(mat(**object_dict))

    # Meshes 
    with open(meshes_json_path, 'r') as f:
        meshes = json.load(f)

    # Pores
    with open(pores_json_path, 'r') as f:
        pores = json.load(f)

    # Simulations
    with open(simulations_json_path, 'r') as f:
        simulations = json.load(f)

    # Change working directory to temporary folder
    path_temp = config.get('LOCAL', 'path_temp')
    os.chdir(path_temp)

    ###########
    # Prepare #
    ###########

    simulation_data = prepare_simulations(simulations, pores)

    ###################
    # Run Simulations #
    ###################

    for name, simu_dict in simulation_data.iteritems():
        print >> sys.__stdout__, 'Starting {} simulation(s)'.format(len(simu_dict['Parameters']))
        for index, parameters in enumerate(simu_dict['Parameters']):
            print >> sys.__stdout__, 'Run simulation {}/{} ...'.format(index+1, len(simu_dict['Parameters']))
            try:
                name_index = '{}_{}'.format(name, index)
                p = MetamaterialFinder(name_index)
                p.create_materials(materials)
                pores = []
                for pf, param, mat in parameters:
                    coordinates = getattr(PoreFunctions, pf)(*param[3:], x_offset=param[0],
                                                              y_offset=param[1], rot_angle=param[2])
                    pores.append({'coordinates': coordinates, 'material': mat})
                print >> sys.__stdout__, '\tCreate sketches ...'
                p.create_sketches(pores, (simu_dict['LX'], simu_dict['LY']),
                                  N=(simu_dict['NX'], simu_dict['NY']),
                                  side_lengths_full=(simu_dict['SX'], simu_dict['SY']),
                                  force_full=simu_dict['Force full'],
                                  force_homogenization=simu_dict['Force homogenization'])
                print >> sys.__stdout__, '\tCreate parts ...'
                p.create_parts(simu_dict['LZ'], dimension=simu_dict['Dimension'])
                print >> sys.__stdout__, '\tCreate mesh ...'
                try:
                    mesh_params = meshes[simu_dict['Mesh name']]
                except:
                    print >> sys.__stdout__, '\t\tMesh specifier not found, aborting ...'
                try:
                    mesh_size_z = mesh_params['mesh_size_z']
                except:
                    mesh_size_z = None
                p.create_mesh(mesh_params['mesh_size'], mesh_params['deviation_factor'],
                              mesh_params['min_size_factor'], mesh_params['element_shape'],
                              mesh_params['element_type'], mesh_size_z)
                print >> sys.__stdout__, '\tCreate sets ...'
                p.create_sets()
                print >> sys.__stdout__, '\tCreate sections ...'
                p.create_sections()
                p.assign_sections(simu_dict['Outside material'])
                print >> sys.__stdout__, '\tCreate assembly ...'
                p.create_assembly(simu_dict['Extensometer'])
                print >> sys.__stdout__, '\tCreate constraints ...'
                p.create_constraints()
                print >> sys.__stdout__, '\tCreate steps and history ...'
                p.create_steps_and_history(nlgeom=simu_dict['Nlgeom'], stepsize=simu_dict['Stepsize'])
                print >> sys.__stdout__, '\tCreate amplitude ...'
                p.create_amplitude()
                print >> sys.__stdout__, '\tCreate boundaries ...'
                p.create_boundaries(simu_dict['UY'])
                print >> sys.__stdout__, '\tSave model ...'
                p.save_model(path_temp)
                print >> sys.__stdout__, '\tRun simulation ...'
                try:
                    p.run_model()
                    p.post_process(path_temp, simu_dict['Save directory'], parameters)
                    print >> sys.__stdout__, '\tSimulation finished successfully ...'
                except OdbError:
                    print >> sys.__stdout__, 'Error in simulation (aborted).'
            except:
                print >> sys.__stdout__, 'Error in simulation (aborted).'