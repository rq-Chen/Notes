# Graph and Network Theory

## Tessellation

### Voronoi Diagram

- Also called *Dirichlet tessellation*.
- Given a set of *N* points, divide the plane into *N* cells, each containing one point and the area that is closest to this point (rather than other *N-1* points).

### Delaunay Triangulation

- Given a set of *N* points, connect them into triangles so that no point is inside the circumcircle of any triangle.
- It maximizes the minimum angle in the whole triangulation, so it tends to avoid large obtuse angles.
- *Delaunay triangulation* corresponds with the *dual graph* of Voronoi diagram (connecting the *circumcenters* of adjacent triangles generates the Voronoi diagram).