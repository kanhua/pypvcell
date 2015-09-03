__author__ = 'kanhua'
from iii_v_si import calc_2j_si_iv,calc_3j_si_eta


# State-of-the-art 2J
eta,si_voc,top_voc=calc_2j_si_iv(1000*1e-6, si_rad_eta=5e-3, top_cell_bg=1.7, top_cell_qe=1, top_cell_rad_eta=1.9e-3)

print(eta,top_voc,si_voc)

# imporved state-of-the-art 2J
eta,si_voc,top_voc=calc_2j_si_iv(1000*1e-6, si_rad_eta=5e-3, top_cell_bg=1.7, top_cell_qe=1, top_cell_rad_eta=1e-2)

print(eta,top_voc,si_voc)


# State-of-the-art 3J
eta,top_voc,mid_voc,bot_voc=calc_3j_si_eta(1e-4,5e-3,1,top_band_gap=1.97,mid_band_gap=1.48)

print(eta,top_voc,mid_voc,bot_voc)

# improved state-of-the-art 3J
eta,top_voc,mid_voc,bot_voc=calc_3j_si_eta(1e-3,5e-3,1,top_band_gap=1.97,mid_band_gap=1.48)

print(eta,top_voc,mid_voc,bot_voc)






