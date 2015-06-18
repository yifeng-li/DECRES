import cisreg
import numpy as np
# the color
rgb_codes={"EnhancerActive":"255,215,0", #gold, EnhancerActive
              "EnhancerInactive":"184,134,11", # dark golden rod, EnhancerInactive
              "PromoterActive":"255,0,0", # red, PromoterActive
              "PromoterInactive":"250,128,114", # salmon, PromoterInactive
              "Exon":"0,128,0", # green, Exon
              "Unknown":"128,128,128",# gray, Unknown
              "EnhancerInactive+PromoterInactive+Exon+Unknown":"128,128,128"} # EnhancerInactive+PromoterInactive+Exon+Unknown

regions=[["chr1","100","110","Enhancer",".",".",".",".","."],
            ["chr1","110","120","Enhancer",".",".",".",".","."],
            ["chr1","120","130","Enhancer",".",".",".",".","."],
            ["chr1","130","140","Promoter",".",".",".",".","."],
            ["chr1","140","150","Enhancer",".",".",".",".","."],
            ["chr1","150","160","Unknown",".",".",".",".","."],
            ["chr2","160","170","Promoter",".",".",".",".","."]]
regions=np.array(regions)
path='/home/yifengli/prog/my/test'
regionnames=["EnhancerActive","EnhancerActive","EnhancerActive","PromoterActive","EnhancerActive","EnhancerInactive+PromoterInactive+Exon+Unknown","PromoterActive"]
regionnames=np.array(regionnames)
filename="CELL"
cisreg.write_bed(path,filename,regionnames,rgb_codes,regions,merge=True,background="EnhancerInactive+PromoterInactive+Exon+Unknown")
