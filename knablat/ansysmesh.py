"""
import ansys mesh(.msh file default used by Ansys Fluent) to knablat mesh

this is modified from meshio/ansys/_ansys.py

I/O for Ansys's msh format.

<https://romeo.univ-reims.fr/documents/fluent/tgrid/ug/appb.pdf>
"""
import re

import numpy as np

from .mesh import Face, Cell, Mesh, Boundary, CellBlock


class ReadError(Exception):
    pass


def _skip_to(f, char):
    c = None
    while c != char:
        c = f.read(1).decode()


def _skip_close(f, num_open_brackets):
    while num_open_brackets > 0:
        char = f.read(1).decode()
        if char == "(":
            num_open_brackets += 1
        elif char == ")":
            num_open_brackets -= 1


def _read_points(f, line, first_point_index_overall, last_point_index):
    # If the line is self-contained, it is merely a declaration
    # of the total number of points.
    if line.count("(") == line.count(")"):
        return None, None, None

    # (3010 (zone-id first-index last-index type ND)
    out = re.match("\\s*\\(\\s*(|20|30)10\\s*\\(([^\\)]*)\\).*", line)
    assert out is not None
    a = [int(num, 16) for num in out.group(2).split()]

    if len(a) <= 4:
        raise ReadError()

    first_point_index = a[1]
    # store the very first point index
    if first_point_index_overall is None:
        first_point_index_overall = first_point_index
    # make sure that point arrays are subsequent
    if last_point_index is not None:
        if last_point_index + 1 != first_point_index:
            raise ReadError()
    last_point_index = a[2]
    num_points = last_point_index - first_point_index + 1
    dim = a[4]

    # Skip ahead to the byte that opens the data block (might
    # be the current line already).
    last_char = line.strip()[-1]
    while last_char != "(":
        last_char = f.read(1).decode()

    if out.group(1) == "":
        # ASCII data
        pts = np.empty((num_points, dim))
        for k in range(num_points):
            # skip ahead to the first line with data
            line = ""
            while line.strip() == "":
                line = f.readline().decode()
            dat = line.split()
            if len(dat) != dim:
                raise ReadError()
            for d in range(dim):
                pts[k][d] = float(dat[d])
    else:
        # binary data
        if out.group(1) == "20":
            dtype = np.float32
        else:
            if out.group(1) != "30":
                ReadError(f"Expected keys '20' or '30', got {out.group(1)}.")
            dtype = np.float64
        # read point data
        pts = np.fromfile(f, count=dim * num_points, dtype=dtype).reshape(
            (num_points, dim)
        )

    # make sure that the data set is properly closed
    _skip_close(f, 2)
    return pts, first_point_index_overall, last_point_index


def _read_cells_declaration(f, line):
    """ read cells declaration section, return the number of cells """
    out = re.match("\\s*\\(\\s*(|20|30)12\\s*\\(([^\\)]+)\\).*", line)
    assert out is not None
    a = [int(num, 16) for num in out.group(2).split()]
    return a[0], a[1], a[2] 


def _read_faces(f, line):
    # faces
    # (13 (zone-id first-index last-index type element-type))

    # If the line is self-contained, it is merely a declaration of
    # the total number of points.
    if line.count("(") == line.count(")"):
        return None, None

    out = re.match("\\s*\\(\\s*(|20|30)13\\s*\\(([^\\)]+)\\).*", line)
    assert out is not None
    a = [int(num, 16) for num in out.group(2).split()]

    if len(a) <= 4:
        raise ReadError()
    zone_id = a[0]
    first_index = a[1]
    last_index = a[2]
    num_cells = last_index - first_index + 1
    element_type = a[4]

    element_type_to_key_num_nodes = {
        0: ("mixed", None),
        2: ("line", 2),
        3: ("triangle", 3),
        4: ("quad", 4),
    }

    key, num_nodes_per_cell = element_type_to_key_num_nodes[element_type]

    # Skip ahead to the line that opens the data block (might be
    # the current line already).
    if line.strip()[-1] != "(":
        _skip_to(f, "(")

    data = []
    if out.group(1) == "":
        # ASCII
        if key == "mixed":
            # From
            # <https://www.afs.enea.it/project/neptunius/docs/fluent/html/ug/node1471.htm>:
            #
            # > If the face zone is of mixed type (element-type = > 0), the body of the
            # > section will include the face type and will appear as follows
            # >
            # > type v0 v1 v2 c0 c1
            # >
            for k in range(num_cells):
                line = ""
                while line.strip() == "":
                    line = f.readline().decode()
                dat = line.split()
                type_index = int(dat[0], 16)
                if type_index == 0:
                    raise ReadError()
                type_string, num_nodes_per_cell = element_type_to_key_num_nodes[
                    type_index
                ]
                if len(dat) != num_nodes_per_cell + 3:
                    raise ReadError()

                data.append([int(d, 16) for d in dat[1:]])

            data = np.array(data)

        else:
            # read cell data
            data = np.empty((num_cells, num_nodes_per_cell), dtype=int)
            for k in range(num_cells):
                line = f.readline().decode()
                dat = line.split()
                # The body of a regular face section contains the grid connectivity, and
                # each line appears as follows:
                #   n0 n1 n2 cr cl
                # where n* are the defining nodes (vertices) of the face, and c* are the
                # adjacent cells.
                if len(dat) != num_nodes_per_cell + 2:
                    raise ReadError()
                data[k] = [int(d, 16) for d in dat]
    else:
        # binary
        if out.group(1) == "20":
            dtype = np.int32
        else:
            if out.group(1) != "30":
                ReadError(f"Expected keys '20' or '30', got {out.group(1)}.")
            dtype = np.int64

        if key == "mixed":
            raise ReadError("Mixed element type for binary faces not supported yet")

        # Read cell data.
        # The body of a regular face section contains the grid
        # connectivity, and each line appears as follows:
        #   n0 n1 n2 cr cl
        # where n* are the defining nodes (vertices) of the face,
        # and c* are the adjacent cells.
        shape = (num_cells, num_nodes_per_cell + 2)
        count = shape[0] * shape[1]
        data = np.fromfile(f, count=count, dtype=dtype).reshape(shape)

    # make sure that the data set is properly closed
    _skip_close(f, 2)

    return zone_id, data


def _read_zone_physics(f, line):
    """ read zone physics """
    # (45 (2 fluid solid)())
    out = re.match("\\s*\\(\\s*45\\s*\\(([^\\)]+)\\).*", line)
    assert out is not None
    a = out.group(1).split()
    return int(a[0]), a[1], a[2]


def read(filename):  # noqa: C901
    # Initialize the data optional data fields
    field_data = {}
    cell_data = {}
    point_data = {}

    points = []
    faces = {}
    # cells = []
    cell_zones = {}
    physics = []
    first_point_index_overall = None # first point index is one according to the document
    last_point_index = None
    cell_number = None

    # read file in binary mode since some data might be binary
    with open(filename, "rb") as f:
        while True:
            line = f.readline().decode()
            if not line:
                break

            if line.strip() == "":
                continue

            # expect the line to have the form
            #  (<index> [...]
            out = re.match("\\s*\\(\\s*([0-9]+).*", line)
            if not out:
                raise ReadError()
            index = out.group(1)

            if index == "0":
                # Comment.
                _skip_close(f, line.count("(") - line.count(")"))
            elif index == "1":
                # header
                # (1 "<text>")
                _skip_close(f, line.count("(") - line.count(")"))
            elif index == "2":
                # dimensionality
                # (2 3)
                _skip_close(f, line.count("(") - line.count(")"))
            elif re.match("(|20|30)10", index):
                # points
                pts, first_point_index_overall, last_point_index = _read_points(
                    f, line, first_point_index_overall, last_point_index
                )

                if pts is not None:
                    points.append(pts)

            elif re.match("(|20|30)12", index):
                # cells
                # (2012 (zone-id first-index last-index type element-type))
                if line.count("(") == line.count(")"): # declaration section
                    zone_id, first_index, last_index = _read_cells_declaration(f, line)
                    if zone_id == 0:
                        cell_number = last_index
                    else:
                        # add cell zone
                        if zone_id in cell_zones:
                            raise ReadError(f"Duplicate zone id {zone_id}.")
                        cell_zones[zone_id] = (first_index, last_index)
                # ignore cell body
                else:
                    _skip_close(f, line.count("(") - line.count(")"))
            elif re.match("(|20|30)13", index):
                zone_id, data = _read_faces(f, line)
                if zone_id is None:
                    continue
                if zone_id in faces:
                    raise ReadError(f"Duplicate zone id {zone_id}.")
                faces[zone_id] = data

            elif index == "39":
                print("Warning: Zone specification not supported yet. Skipping.")
                _skip_close(f, line.count("(") - line.count(")"))

            elif index == "45":
                # read zone physics
                # (45 (2 fluid solid)())
                zone_id, physics_type, physics_name = _read_zone_physics(f, line)
                physics.append((zone_id, physics_type, physics_name))
            else:
                print(f"Warning: Unknown index {index}. Skipping.")
                # Skipping ahead to the next line with two closing brackets.
                _skip_close(f, line.count("(") - line.count(")"))

    points = np.concatenate(points)


    # according to the document, the first_point_index is 1, 
    # and face's adjacent cell index is also from 1
    # so we don't need to change the index
    # Gauge the cells with the first point_index.
    # for k, fc in enumerate(faces):
    #     faces[k] = (fc[0], fc[1] - first_point_index_overall)

    return points, faces, cell_zones, physics, cell_number


def read_ans_mesh(filename: str) -> Mesh:
    """ read ansys mesh file and setup Mesh """
    points, faces_, cell_zones, physics, num_cells = read(filename)
    faces = []
    face_zones = {}
    blocks = []
    boundary = {}
    interior_faces = []
    cells = [Cell([], i) for i in range(num_cells)]
    n = 0
    for zone_id, data in faces_.items():
        face_zones[zone_id] = []
        for i, fc in enumerate(data, n):
            # ansys mesh face normal point to owner cell,
            # different from openfoam mesh, reverse the order
            face = Face([points[p-1] for p in fc[-3::-1]], i)
            face.setup()
            faces.append(face)
            face_zones[zone_id].append(face)
            cells[fc[-2] - 1].faces.append(face)
            face.owner = cells[fc[-2] - 1]
            if fc[-1] > 0:
                # interior_faces.append(face)
                cells[fc[-1] - 1].faces.append(face)
                face.neighbour = cells[fc[-1] - 1]
        n = i + 1
    for c in cells:
        c.setup()
    for zone_id, physics_type, physics_name in physics:
        if physics_type == "fluid" or physics_type == "solid":
            if zone_id in cell_zones:
                blocks.append(CellBlock(
                    [cells[i-1] for i in range(cell_zones[zone_id][0], cell_zones[zone_id][1]+1)], 
                    zone_id, physics_type, physics_name))
        elif physics_type == "interior":
            if zone_id in face_zones:
                interior_faces.extend(face_zones[zone_id])
        else:
            if zone_id in face_zones:
                # use physics_name as boundary label
                boundary[physics_name] = Boundary(zone_id, 
                                         face_zones[zone_id],
                                         physics_type, physics_name)
    return Mesh(points, faces, cells, blocks, boundary, interior_faces)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('usage: python -m knablat.ansysmesh <path to mesh file>')
        exit()
    else:
        filename = sys.argv[1]
    mesh = read_ans_mesh(filename)
    mesh.info()
    for k, b in mesh.boundary.items():
        print(k, b.type, b.bid, len(b.faces))
