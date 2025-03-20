import tables
import os
import numpy as np

def save_to_hdf5(
    x_grid,
    y_grid,
    u_grid,
    v_grid,
    v_mag,
    s11,
    s22,
    s33,
    w_vor,
    wall_edges,
    baffle_edges,
    Normalized,
    paths,
):
    hdf5_path = os.path.join(paths["Results"], "metadata_features.hdf5")
    TABLE_FILTERS = tables.Filters(
        complevel=5, complib="zlib", shuffle=True, fletcher32=True
    )
    print("Start: saving the data")

    with tables.File(hdf5_path, mode="w") as f:
        # Save 2D arrays
        f.create_carray(f.root, "x_grid", obj=x_grid, filters=TABLE_FILTERS)
        f.create_carray(f.root, "y_grid", obj=y_grid, filters=TABLE_FILTERS)
        f.create_carray(f.root, "wall_edges", obj=wall_edges, filters=TABLE_FILTERS)
        f.create_carray(f.root, "baffle_edges", obj=baffle_edges, filters=TABLE_FILTERS)

        # Save 3D arrays
        f.create_carray(f.root, "u_grid", obj=u_grid, filters=TABLE_FILTERS)
        f.create_carray(f.root, "v_grid", obj=v_grid, filters=TABLE_FILTERS)
        f.create_carray(f.root, "v_mag", obj=v_mag, filters=TABLE_FILTERS)
        f.create_carray(f.root, "s11", obj=s11, filters=TABLE_FILTERS)
        f.create_carray(f.root, "s22", obj=s22, filters=TABLE_FILTERS)
        f.create_carray(f.root, "tau_max", obj=s33, filters=TABLE_FILTERS)
        f.create_carray(f.root, "w_vor", obj=w_vor, filters=TABLE_FILTERS)
        
        # Save boolean value
        f.create_array(f.root, "Normalised_data", obj=np.string_(str(Normalized)))

        print("Finish: saving the data")
