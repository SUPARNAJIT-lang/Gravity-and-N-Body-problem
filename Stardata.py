#Position calcn✅✅✅:
def pcalc():
    from astroquery.simbad import Simbad
    from astropy import units as u 
    from astropy.coordinates import SkyCoord


    customsimbad=Simbad()
    customsimbad.add_votable_fields('plx')

    stars=["Alpha Cen A", "Alpha Cen B", "Proxima Cen"]


    result=customsimbad.query_objects(stars)
    return result




# In cartesian coordinate system:


import numpy as np

#α centauri A


ra=219.90205833170774
dec=-60.83399268831004
plx_value=742.12


γ=np.pi/180

def result_crtsn(m,n,r):
    plxarc=r/1000
    d=(1/plxarc)*3.086e16  #in m

    x=d*(np.cos(γ*n))*np.cos(γ*m)
    y=d*(np.cos(γ*n))*np.sin(γ*m)
    z=d*np.sin(γ*m)

    return (x,y,z)

ϙ=result_crtsn(ra,dec,plx_value)


# Actual velocity calcn (km/s)✅✅✅:

from astroquery.simbad import Simbad
import numpy as np

# 1️⃣ Query Simbad with correct field names
custom = Simbad()
custom.add_votable_fields('ra', 'dec', 'plx_value', 'pmra', 'pmdec', 'rvz_radvel')

stars = ["Alpha Cen A", "Alpha Cen B", "Proxima Cen"]
res = custom.query_objects(stars)

# 2️⃣ Convert column names to lowercase for consistency
res.rename_columns(res.colnames, [c.lower() for c in res.colnames])

# 3️⃣ Conversion constant: mas/yr * pc -> km/s
k = 4.74047

def compute_velocity_array(row):
    # Read values
    ra = np.deg2rad(row['ra'])
    dec = np.deg2rad(row['dec'])
    plx = row['plx_value']  # in mas
    # Divide by 1000 to convert mas/yr -> arcsec/yr
    pmra = (row['pmra'] if row['pmra'] is not None else 0.0) / 1000
    pmdec = (row['pmdec'] if row['pmdec'] is not None else 0.0) / 1000
    rv = row['rvz_radvel'] if row['rvz_radvel'] is not None else 0.0  # km/s

    if plx == 0 or plx is None:
        raise ValueError(f"Parallax missing for {row['main_id']}")

    # Distance in parsec
    d = 1 / (plx * 1e-3)  # mas -> arcsec -> pc

    # Unit vectors for ICRS
    cosd = np.cos(dec)
    sind = np.sin(dec)
    cosa = np.cos(ra)
    sina = np.sin(ra)

    e_r = np.array([cosd * cosa, cosd * sina, sind])
    e_alpha = np.array([-sina, cosa, 0.0])
    e_delta = np.array([-sind * cosa, -sind * sina, cosd])

    # Tangential velocity (km/s)
    v_tan = k * d * (pmra * e_alpha + pmdec * e_delta)
    # Radial velocity (km/s)
    v_rad = rv * e_r
    # Total space velocity
    v_total = v_tan + v_rad

    return v_total  # numpy array [vx, vy, vz] in km/s

# 4️⃣ Compute velocities
velocity_arrays = []
for r in res:
    try:
        velocity_arrays.append(compute_velocity_array(r))
    except Exception as e:
        print(f"Skipping {r['main_id']}: {e}")

# 5️⃣ Print results
for star, v in zip(stars, velocity_arrays):
    print(f"{star}: vx, vy, vz = {v}")
