"""
A module for cis-regulatory elements processing.

Funtionality include:

Yifeng Li
CMMT, UBC, Vancouver
Dec 11, 2014
Contact: yifeng.li.cn@gmail.com
"""
from __future__ import division
import numpy as np
import math
import os

def write_bed(path,filename,regionnames,regionsocres,rgb_codes,regions,merge=True,background=""):
    """
    write the predicted labels to a bed file.
    path: string.
    filename: string, bed file name, and track name, e.g. "GM12878_GM12878".
    regionnames: string 1d numpy array, the predicted class labels.
    regionscores: 1d numpy array of type int, the scores (1000*prob) of the prdicted class labels. It can be [] for using the original score in the regions.
    regions: 2d array, the regions.
    merged: bool, if the regions need to be merged based the region names.
    Yifeng Li, Dec 11, 2014
    """
    try:
        os.makedirs(path)
    except OSError:
        pass
    # write the head
    print "writing the bed head"
    line1="browser position chr1:0-1000\n"
    line2="browser dense " + filename + "\n"
    line3="track name=\"" + filename + "\" description=\"" + filename + "\" visibility=3 itemRgb=\"On\" useScore=0 db=hg19\n"
    filename=path + "/" +filename + ".bed"
    f=open(filename,'w')
    f.write(line1)
    f.write(line2)
    f.write(line3)
    f.close()
    # the region name
    regions=np.array(regions)
    regions[:,3]=regionnames
    # if the scores of the regions is not empty, then assign it
    if len(regionsocres)>0:
        regions[:,4]=regionsocres
    
    # the color
    #rgb_codes={"EnhancerActive":"255,215,0", #gold, EnhancerActive
    #          "EnhancerInactive":"184,134,11", # dark golden rod, EnhancerInactive
    #          "PromoterActive":"255,0,0", # red, PromoterActive
    #          "PromoterInactive":"250,128,114", # salmon, PromoterInactive
    #          "Exon":"0,128,0", # green, Exon
    #          "Unknown":"128,128,128",# gray, Unknown
    #          "EnhancerInactive+PromoterInactive+Exon+Unknown":"128,128,128"} # EnhancerInactive+PromoterInactive+Exon+Unknown
    nr,nc=regions.shape
    if nc >= 9:
        get_color_for_elements(regionnames,rgb_codes) # change regionnames to colors
        regions[:,8]=regionnames
    
    if merge:
        print "Need to merge regions"
        regions=merge_regions(regions,d=0,background=background)

    print "Saving in bed file"
    file_handle=file(filename,'a')
    np.savetxt(file_handle,regions,fmt="%s",delimiter="\t")
    file_handle.close()

def get_color_for_elements(regionnames,rgb_codes):
    for ele in rgb_codes:
        print ele
        regionnames[regionnames==ele]=rgb_codes[ele]
    
def merge_regions(regions,d=0,background=""):
    """
    merge regions based on region names.
    regions: nympy 2d array, the regions to be merged.
    Yifeng Li, Jan 11, 2015

    Example:
    regions=[["chr1","100","110","Enhancer"],
            ["chr1","110","120","Enhancer"],
            ["chr1","120","130","Enhancer"],
            ["chr1","130","140","Promoter"],
            ["chr1","140","150","Enhancer"],
            ["chr1","150","160","Promoter"],
            ["chr2","160","170","Promoter"]]
    regions=np.array(regions)
    M=merge_regions(regions)
    """

    print "Starting merging regions ..."
    nr,nc=regions.shape
    ind=[] #np.array([],dtype=int)
    r=0 # index to be added in ind, the beginning of a merged region
    while r<=nr-1:
        if regions[r,3]==background: # if this element is a background, skip
            r=r+1
            continue
        ind.append(r)
        if r is nr-1:
            break
        j=r # the end of a merged region
        #                  same chr                        same name                        overlap
        while j+1<=nr-1 and regions[j,0]==regions[j+1,0] and regions[j,3]==regions[j+1,3] and int(regions[j+1,1])<=int(regions[j,2])+d:
            j=j+1
        if j>r:
            regions[r,2]=regions[j,2] # end
            if nc==9:
                regions[r,7]=regions[j,7] # the 8th field
        r=j+1
    regions=regions[ind,:]
    print "Finished merging regions"
    return regions

def sample_bed(bed_in,bed_out=None,number=None,percent=None,score_min=0,score_max=1000,reg_width_min=None,rng=np.random.RandomState(100)):
    """
    Sample a subset from the bed file.
    bed_in: either a string indicating the file name, or a numpy 2d array.
    bed_out: if it is None: return a 2d array; if it is a string: the name of the file to save the sampled bed.
    number: int, number of regions to be sampled.
    percent: float, percentage of regions to be sampled.
    """
    # load bed
    if isinstance(bed_in,str):
        bed_in=np.loadtxt(bed_in,dtype=object)

    # get indices
    num_reg=len(bed_in)
    indices_range=np.array(range(num_reg))
    scores=np.array(bed_in[:,4],dtype=int)
    if reg_width_min is None:
        log_satisfied=np.logical_and(scores>=score_min, scores<=score_max)
    else:
        widths=np.array(bed_in[:,2],dtype=int)-np.array(bed_in[:,1],dtype=int)
        log_satisfied=np.logical_and( np.logical_and(scores>=score_min, scores<=score_max), widths>=reg_width_min )
        print np.sum(log_satisfied)
    indices_satisfied=indices_range[log_satisfied]
    num_satisfied=len(indices_satisfied)

    # sample
    if percent is not None:
        number = int(round(percent*num_reg))
    if number is not None:
        if number > num_satisfied:
            number=num_satisfied
    indices_sampled=indices_satisfied[rng.choice(num_satisfied,size=number,replace=False)]
    indices_sampled.sort()

    # save bed
    if isinstance(bed_out,str):
        np.savetxt(bed_out,bed_in[indices_sampled,:],fmt='%s',delimiter='\t')
    else:
        return bed_in[indices_sampled,:]

def get_exons_from_RefSeqGene(filename_genes,filename_exons,chr=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']):
    """
    extract exons from the RefSeq Genes. The result is not sorted, not unique. sort and uniq should be used, after using this function.
    """
    # load data
    genes=np.loadtxt(filename_genes,dtype=object,delimiter='\t',skiprows=1)
    chr=np.array(chr)

    # extract exons
    exons=[]
    for gene in genes:
        chrom=gene[2]
        if np.sum(chrom==chr)==0: # not in the chr list
            continue
        strand=gene[3]
        exon_starts=gene[9].split(',')
        exon_ends=gene[10].split(',')
        exon_starts=exon_starts[0:-1]
        exon_ends=exon_ends[0:-1]
        #exon_starts=np.array(exon_starts.split(','),dtype=int)
        #exon_ends=np.array(exon_ends.split(','),dtype=int)
        name=gene[12]
        count=0
        for exon_start,exon_end in zip(exon_starts,exon_ends):
            count=count+1
            exons.append([chrom,exon_start,exon_end,name,0,strand])
    exons=np.array(exons,dtype=object)
    # sort and unique: use unix sort and uniq
    # save
    np.savetxt(filename_exons,exons,fmt='%s',delimiter='\t')

def get_genes_from_RefSeqGene(filename_genes,filename_genes_plus,filename_genes_minus,chr=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']):
    """
    extract gene loci from the RefSeq Genes. The outputs are the forward and reverse loci in two different bedGraph files. The result is not sorted, not unique. After using this function, sort and uniq should be used. Finally bedtools merge should be used.
    """
    # load data
    genes=np.loadtxt(filename_genes,dtype=object,delimiter='\t',skiprows=1)
    chr=np.array(chr)

    # extract genes
    genes_plus=[]
    genes_minus=[]
    for gene in genes:
        chrom=gene[2]
        if np.sum(chrom==chr)==0: # not in the chr list
            continue
        strand=gene[3]
        gene_start=gene[4]
        gene_end=gene[5]
        gene_name=gene[12]
        if strand=='+':
            genes_plus.append([chrom,gene_start,gene_end,1])
        else:
            genes_minus.append([chrom,gene_start,gene_end,1])
    genes_plus=np.array(genes_plus,dtype=object)
    genes_minus=np.array(genes_minus,dtype=object)
    # sort and unique: use unix sort and uniq
    # merge use bedtools merge
    # save
    np.savetxt(filename_genes_plus,genes_plus,fmt='%s',delimiter='\t')
    np.savetxt(filename_genes_minus,genes_minus,fmt='%s',delimiter='\t')

    


    
