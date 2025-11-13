#Position calcn✅✅✅:

def pcalc1():
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
α=(219.90205833170774,-60.83399268831004,742.12)

#α centauri B
β=(219.89609628987276,-60.83752756558407,742.12)

#Proxima centauri 
proxima=(217.42894222160578,-62.67949018907555,768.0665)




γ=np.pi/180

def result_crtsn(m,n,r):
    plxarc=r/1000
    d=(1/plxarc)*3.086e16  #in m

    x=d*(np.cos(γ*n))*np.cos(γ*m)
    y=d*(np.cos(γ*n))*np.sin(γ*m)
    z=d*np.sin(γ*m)

    return (x,y,z)

def pcalc2():
    ϙ1=result_crtsn(α[0],α[1],α[2])
    ϙ2=result_crtsn(β[0],β[1],β[2])
    ϙ3=result_crtsn(proxima[0],proxima[1],proxima[2])

    return np.array([ϙ1,ϙ2,ϙ3]) 


    





# Actual velocity calcn (m/s)✅✅✅:


from astroquery.simbad import Simbad
import numpy as np

def get_velocity_arrays(stars=None):
    if stars is None:
        stars = ["Alpha Cen A", "Alpha Cen B", "Proxima Cen"]

    # Query Simbad
    custom = Simbad()
    custom.add_votable_fields('ra', 'dec', 'plx_value', 'pmra', 'pmdec', 'rvz_radvel')
    res = custom.query_objects(stars)
    res.rename_columns(res.colnames, [c.lower() for c in res.colnames])
    
    # Conversion constant: mas/yr * pc -> km/s
    k = 4.74047

    def compute_velocity_array(row):
        ra = np.deg2rad(row['ra'])
        dec = np.deg2rad(row['dec'])
        plx = row['plx_value']  # in mas
        pmra = (row['pmra'] if row['pmra'] is not None else 0.0) / 1000
        pmdec = (row['pmdec'] if row['pmdec'] is not None else 0.0) / 1000
        rv = row['rvz_radvel'] if row['rvz_radvel'] is not None else 0.0

        if plx == 0 or plx is None:
            raise ValueError(f"Parallax missing for {row['main_id']}")

        # Distance in parsec
        d = 1 / (plx * 1e-3)

        # Unit vectors for ICRS
        cosd, sind, cosa, sina = np.cos(dec), np.sin(dec), np.cos(ra), np.sin(ra)
        e_r = np.array([cosd*cosa, cosd*sina, sind])
        e_alpha = np.array([-sina, cosa, 0.0])
        e_delta = np.array([-sind*cosa, -sind*sina, cosd])

        # Velocities
        v_tan = k * d * (pmra*e_alpha + pmdec*e_delta)
        v_rad = rv * e_r
        return v_tan + v_rad

    # Compute velocities and store in a NumPy array
    velocity_arrays = []
    for r in res:
        try:
            velocity_arrays.append(compute_velocity_array(r))
        except Exception as e:
            print(f"Skipping {r['main_id']}: {e}")

    # Convert list of arrays into a single NumPy array
    return np.array(velocity_arrays)*1000  # shape: (N_stars, 3)

