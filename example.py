"""
Example file to demonstrate a full conversion run on the included example data.
"""


import bsim2ekv as b2e
import ekv

filenamedict = {
    "wl_is_temp_data" : "./exampledata/0u18_nmos/0u18_ne_wl_is_temp.vcsv",
    "wl_vp_vg_temp_data" : "./exampledata/0u18_nmos/0u18_ne_wl_vp_vg_tempdata.vcsv",
    "wl_id_vg_vb_linear_data" : "./exampledata/0u18_nmos/0u18_ne_wl_id_vg_vb.vcsv",
    "wl_id_vg_linear_temp_data" : "./exampledata/0u18_nmos/0u18_ne_wl_id_vg_linear_temp.vcsv",
    "wl_ib_vg_sat_data" : "./exampledata/0u18_nmos/0u18_ne_wl_saturation_vg_ib.vcsv",
    "wl_noise_data" : "./exampledata/0u18_nmos/0u18_ne_wl_id_sat_noise.vcsv",
    "ws_is_data" : "./exampledata/0u18_nmos/0u18_ne_ws_is_temp.vcsv",
    "ws_vp_vg_data" : "./exampledata/0u18_nmos/0u18_ne_ws_set_vp_vg.vcsv",
    "ws_id_vg_linear_data" : "./exampledata/0u18_nmos/0u18_ne_ws_linear_region_id.vcsv",
    "ws_id_vg_sat_temp_data" : "./exampledata/0u18_nmos/0u18_ne_ws_id_vg_sat_temp.vcsv",
    "ws_id_vd_sat_data" : "./exampledata/0u18_nmos/0u18_ne_ws_saturation_vd_id.vcsv",
    "nl_is_data" : "./exampledata/0u18_nmos/0u18_ne_nl_is.vcsv",
    "nl_vp_vg_data" : "./exampledata/0u18_nmos/0u18_ne_nl_set_vp_vg.vcsv",
    "sat_region_len_isdata" : "./exampledata/0u18_nmos/0u18_ne_saturation_len_id.vcsv"
}

b2e.global_gmin = 1e-12
b2e.global_lmin = 180e-9

print(f'Global gmin set to {b2e.global_gmin}')
print(f'Global lmin set to {b2e.global_lmin}')

datalump = b2e.data_ingest(filenamedict)

print('Data imported into datalump')

initparams = {
    'TNOM' : 300,
    'TOX' : 4e-9,
    'NSUB' : 8.45e17,
    'XJ' : 150e-9,
    'DW' : -2*11.679e-9,
    'DL' : -2*13.267e-9,
    'U0' : 12.74093e-3,
    'VMAX' : 93.964e3,
    'IBB' : 434426950.2685,
    'IBA' : 5415957346.3027,
}

testMOS = ekv.MOS(initparams)

print('testMOS initialized')

print('Commencing full-run. This can take 2h or more with a fairly powerful CPU.')

b2e.full_run(
    datalump,
    testMOS,
    proclist=[
        'vto_phi_gamma',
        'leta',
        'weta',
        'lk_q0',
        'kp_e0',
        'kp_e0',
        'dl',
        'lambda_ucrit',
        'lambda',
        'iba_ibb_ibn',
        'tcv',
        'bex',
        'ucex',
        'kf_af'
    ]
)
