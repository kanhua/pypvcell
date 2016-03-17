from detail_balanced_MJ import rad_to_voc, rad_to_voc_fast
from photocurrent import gen_square_qe

test_qe = gen_square_qe(1.12, 0.8)

r1 = rad_to_voc(0.001, test_qe)

r2 = rad_to_voc_fast(0.001, test_qe)

print(r1)
print(r2)
