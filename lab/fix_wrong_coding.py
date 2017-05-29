

test_file='../pypvcell/matdata/nk_TiO2.csv'
with open(test_file,'rb') as fp:
    f_line=fp.read()
    test_str=f_line.decode("utf-8", "strict")
    test_str.replace(r'\r',r'\n')


fp=open(test_file,'w')
fp.write(test_str)
fp.close()

