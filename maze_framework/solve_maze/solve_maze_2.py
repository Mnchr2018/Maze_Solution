from pathlib import Path
import os
import cv2
import pandas as pd
import numpy as np
import random
from statistics import mean

def convert_to_binary(df,threshold):
    return pd.DataFrame(np.where(df>=threshold, 1, 0))

def crop_df(df):
    rows, cols = df.shape
    
    top_line_val=sorted([df.loc[i,:].sum() for i in range(0,int(df.shape[0]/2))])[0]
    bottom_line_val=sorted([df.loc[i,:].sum() for i in range(df.shape[0]-1,int(df.shape[0]/2),-1)])[0]
    left_line_val=sorted([df.loc[:,i].sum() for i in range(0,int(df.shape[1]/2))])[0]
    right_line_val=sorted([df.loc[:,i].sum() for i in range(df.shape[1]-1,int(df.shape[1]/2),-1)])[0]

    rows_to_crop=[]
    cols_to_crop=[]

    for i in range (0,rows):
        if df.loc[i,:].sum()!=top_line_val:
            rows_to_crop.append(i)
        else:
            break
    for i in range(rows, 0, -1):
        if df.loc[i-1,:].sum()!=bottom_line_val:
            rows_to_crop.append(i-1)
        else:
            break
    for i in range (0,cols):
        if df.loc[:,i].sum()!=left_line_val:
            cols_to_crop.append(i)
        else:
            break
    for i in range(cols, 0, -1):
        if df.loc[:,i-1].sum()!=right_line_val:
            cols_to_crop.append(i-1)
        else:
            break
    
    df=df.drop(rows_to_crop,axis=0)
    df=df.drop(cols_to_crop,axis=1)
    df=df.reset_index(drop=True)
    df.columns = range(df.shape[1])
    
    return df

def find_neighbour(data,ij):
    
    i=ij[0]
    j=ij[1]
    
    rows=data.shape[0]
    cols=data.shape[1]

    if i==0:
        if j==0:
            u=None
            d={"val":data.loc[i+1,j], "loc":[i+1,j]}
            r={"val":data.loc[i,j+1], "loc":[i,j+1]}
            l=None
        elif j==cols-1:
            u=None
            d={"val":data.loc[i+1,j], "loc":[i+1,j]}
            r=None
            l={"val":data.loc[i,j-1], "loc":[i,j-1]}
        else:
            u=None
            d={"val":data.loc[i+1,j], "loc":[i+1,j]}
            r={"val":data.loc[i,j+1], "loc":[i,j+1]}
            l={"val":data.loc[i,j-1], "loc":[i,j-1]}
    elif i==rows-1:
        if j==0:
            u={"val": data.loc[i-1,j], "loc":[i-1,j]}
            d=None
            r={"val":data.loc[i,j+1], "loc":[i,j+1]}
            l=None
        elif j==cols-1:
            u={"val": data.loc[i-1,j], "loc":[i-1,j]}
            d=None
            r=None
            l={"val":data.loc[i,j-1], "loc":[i,j-1]}
        else:
            u={"val": data.loc[i-1,j], "loc":[i-1,j]}
            d=None
            r={"val":data.loc[i,j+1], "loc":[i,j+1]}
            l={"val":data.loc[i,j-1], "loc":[i,j-1]}
    
    else:
        if j==0:
            u={"val": data.loc[i-1,j], "loc":[i-1,j]}
            d={"val":data.loc[i+1,j], "loc":[i+1,j]}
            r={"val":data.loc[i,j+1], "loc":[i,j+1]}
            l=None
        elif j==cols-1:
            u={"val": data.loc[i-1,j], "loc":[i-1,j]}
            d={"val":data.loc[i+1,j], "loc":[i+1,j]}
            r=None
            l={"val":data.loc[i,j-1], "loc":[i,j-1]}
        else:
            u={"val": data.loc[i-1,j], "loc":[i-1,j]}
            d={"val":data.loc[i+1,j], "loc":[i+1,j]}
            r={"val":data.loc[i,j+1], "loc":[i,j+1]}
            l={"val":data.loc[i,j-1], "loc":[i,j-1]}
    
    return u,d,r,l

def get_loc_val(x):
    if x!=None:
        return x["loc"],x["val"]
    else:
        return None,None

def remove_dups(aa):
    dedup_a=[aa[0]]
    for i in range(1,len(aa)):
        flag=True
        for j in range(0,len(dedup_a)):
            if dedup_a[j][0]==aa[i][0] and dedup_a[j][1]==aa[i][1]:
                flag=False
        if flag==True:
            dedup_a.append(aa[i])
    return dedup_a

def get_me_color(x):
    if x==0:
        return "background-color: white"
    if x==1:
        return "background-color: red"
    if x==2:
        return "background-color: yellow"
    if x==3:
        return "background-color: yellow"

def solve_maze(data):
    # find location of 2 and 3
    for i in range(0,data.shape[0]):
        for j in range(0,data.shape[1]):
            if data.loc[i,j]==2:
                loc_start=[i,j]
            if data.loc[i,j]==3:
                loc_end=[i,j]
    start_list=[get_loc_val(x)[0] for x in find_neighbour(data,loc_start) if get_loc_val(x)[1]==1]
    searched_ones=[]
    searched_ones.extend(start_list)
    temp_list=start_list
    not_found_flag=True
    while len(temp_list)!=0:
        temp_list=remove_dups([item for sublist in [[get_loc_val(x)[0] for x in find_neighbour(data,item) if get_loc_val(x)[1]==1] for item in temp_list] for item in sublist])
        delta_list=[item for item in temp_list if item not in searched_ones]
        searched_ones.extend(delta_list)
        temp_list=delta_list
        if loc_end in [item for sublist in [[get_loc_val(x)[0] for x in find_neighbour(data,item)] for item in temp_list] for item in sublist]:
            print("Eureka: 3 has been found")
            not_found_flag=False
            break
    if not_found_flag:
        print("mmmm cant find 3")
        success_flag=False
        searched_ones=[]
    else:
        success_flag=True
        
    return loc_start, loc_end, success_flag, searched_ones

def find_segments(my_list):
    count =0
    starting_point=0
    segments=list()
    for i in my_list:
        count=0
        for j in range(starting_point,len(my_list)-1):
            if my_list[j+1]-1==my_list[j]:
                count=count+1
            else:
                starting_point=j+1
                segments.append(count+1)
                break
    return segments

def master_solver(image, thershold_1, thershold_2, thershold_3):
    # read image and convert to a pd dataframe
    img = cv2.imread(image,0)
    img = cv2.GaussianBlur(img, (15,15),0)
    im = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    df=pd.DataFrame(im)

    # Average colour
    avg_col=int(mean([item for sublist in [list(df.loc[i,:]) for i in range(df.shape[0])] for item in sublist]))

    # convert dateframe to binary dataframe
    df=convert_to_binary(df,avg_col)

    # indecis to drop
    index_to_drop=list()
    for i in range(0,df.shape[0]):
        if df.loc[i,:].sum()==df.shape[1]:
            index_to_drop.append(i)
    df=df.drop(index_to_drop)
    df=df.reset_index(drop=True)

    index_to_drop=list()
    for i in range(0,df.shape[1]):
        if df.loc[:,i].sum()==df.shape[0]:
            index_to_drop.append(i)
    df=df.drop(index_to_drop,axis=1)
    df.columns = range(df.shape[1])

    # line and block thickness
    zero_segments_hrz=[[i for i, x in enumerate(df.loc[i,:].tolist()) if x==0] for i in range(df.shape[0])]
    zero_segments_vr=[[i for i, x in enumerate(df.loc[:,i].tolist()) if x==0] for i in range(df.shape[1])]
    zero_segments_total=zero_segments_vr+zero_segments_hrz
    line_length=max(set([item for sublist in [find_segments(i) for i in zero_segments_total] for item in sublist]), key = [item for sublist in [find_segments(i) for i in zero_segments_total] for item in sublist].count)

    one_segments_hrz=[[i for i, x in enumerate(df.loc[i,:].tolist()) if x==1] for i in range(df.shape[0])]
    one_segments_vr=[[i for i, x in enumerate(df.loc[:,i].tolist()) if x==1] for i in range(df.shape[1])]
    one_segments_total=one_segments_vr+one_segments_hrz
    block_length=max(set([item for sublist in [find_segments(i) for i in one_segments_total] for item in sublist]), key = [item for sublist in [find_segments(i) for i in one_segments_total] for item in sublist].count)

    # treating raw image
    rows_dict=dict()
    for i in list(range(0, df.shape[0])):
        if df.loc[i,:].sum()<=thershold_1*df.shape[1]:
            rows_dict[i]=df.loc[i,:].sum()
    
    cols_dict=dict()
    for i in list(range(0, df.shape[1])):
        if df.loc[:,i].sum()<=thershold_1*df.shape[0]:
            cols_dict[i]=df.loc[:,i].sum() 

    row_indx=sorted(list(set(list(rows_dict.keys()))))
    count=1
    for i,v in enumerate(row_indx):
        if (i!=0)&(row_indx[i-1]!=row_indx[i]-1):
            count=count+1
    mat_rows=count

    col_indx=sorted(list(set(list(cols_dict.keys()))))
    count=1
    for i,v in enumerate(col_indx):
        if (i!=0)&(col_indx[i-1]!=col_indx[i]-1):
            count=count+1
    mat_cols=count

    rows=mat_rows*line_length+(mat_rows-1)*block_length
    cols=mat_cols*line_length+(mat_cols-1)*block_length
    clean_df=pd.DataFrame(np.zeros(shape=(rows,cols)))
    for col in clean_df.columns:
        clean_df[col]=1
    
    clean_df.loc[0:line_length-1,:]=0
    clean_df.loc[rows-line_length:rows-1,:]=0
    clean_df.loc[:,0:line_length-1]=0
    clean_df.loc[:,cols-line_length:cols-1]=0
    # start and end
    clean_df.loc[0:line_length-1,line_length:line_length+block_length-1]=1
    clean_df.loc[rows-line_length:rows-1,cols-line_length-block_length:cols-line_length-1]=1

    row_index_dict=dict()
    ind_list=list()
    temp_list=list(rows_dict.keys())
    ind_list_all=list()
    for i,v in enumerate(temp_list):
        if (i!=0)&(temp_list[i-1]==temp_list[i]-1):
            ind_list.append(v)
        else:
            ind_list_all.append(ind_list)
            ind_list=list()
    ind_list_all.append(ind_list)
    del ind_list_all[0]
    rows_indx_all=list()
    for item in ind_list_all:
        temp_item=item.copy()
        temp_item.append(item[0]-1)
        rows_indx_all.append(sorted(temp_item))
    rows_indx_all=dict(zip(range(0,len(rows_indx_all)),rows_indx_all))

    col_index_dict=dict()
    ind_list=list()
    temp_list=list(cols_dict.keys())
    ind_list_all=list()
    for i,v in enumerate(temp_list):
        if (i!=0)&(temp_list[i-1]==temp_list[i]-1):
            ind_list.append(v)
        else:
            ind_list_all.append(ind_list)
            ind_list=list()
    ind_list_all.append(ind_list)
    del ind_list_all[0]
    cols_indx_all=list()
    for item in ind_list_all:
        temp_item=item.copy()
        temp_item.append(item[0]-1)
        cols_indx_all.append(sorted(temp_item))
    cols_indx_all=dict(zip(range(0,len(cols_indx_all)),cols_indx_all))

    for i in range(1,mat_rows-1):
        for j in range(mat_cols-1):
            block_df=df.loc[rows_indx_all[i],cols_indx_all[j][-1]+1:cols_indx_all[j][-1]+block_length]
            if block_df.values.sum()/(block_df.shape[0]*block_df.shape[1])<=thershold_2:
                clean_df.loc[(line_length+block_length)*i:(line_length+block_length)*i+line_length-1,  (line_length+block_length)*j:(line_length+block_length)*j+line_length*2+block_length-1]=0

    for i in range(mat_rows-1):
        for j in range(1,mat_cols-1):
            block_df=df.loc[rows_indx_all[i][-1]+1:rows_indx_all[i][-1]+block_length,cols_indx_all[j]]
            if block_df.values.sum()/(block_df.shape[0]*block_df.shape[1])<=thershold_2:
                clean_df.loc[(line_length+block_length)*i:(line_length+block_length)*i+line_length*2+block_length-1,  (line_length+block_length)*j:(line_length+block_length)*j+line_length-1]=0

    # passing processed matrix 
    df=clean_df.copy()

    unique_cols=[]
    for i in list(range(2, df.shape[0]-2)):
        if df.loc[i,:].sum()>=thershold_3*df.shape[1]:
            unique_cols.extend([i for i, x in enumerate(df.loc[i,:].tolist()) if x==0])  
    raw_cols=sorted(list(set(unique_cols)))
    count=1
    for i,v in enumerate(raw_cols):
        if (i!=0)&(raw_cols[i-1]!=raw_cols[i]-1):
            count=count+1
    mat_cols=count-1

    unique_rows=[]
    for i in list(range(2, df.shape[1]-2)):
        if df.loc[:,i].sum()>=thershold_3*df.shape[0]:
            unique_rows.extend([i for i, x in enumerate(df.loc[:,i].tolist()) if x==0])  
    raw_rows=sorted(list(set(unique_rows)))
    count=1
    for i,v in enumerate(raw_rows):
        if (i!=0)&(raw_rows[i-1]!=raw_rows[i]-1):
            count=count+1
    mat_raws=count-1

    pseudo_df=pd.DataFrame(np.zeros(shape=(mat_raws*2+1,mat_cols*2+1)))
    pseudo_df=pd.DataFrame(np.where(pseudo_df==0, 1, 0))

    for i in range(0,mat_raws+1):
        for j in range(0,mat_cols+1):
            box=df.loc[(block_length+line_length)*i:(block_length+line_length)*i+line_length-1,(block_length+line_length)*j:(block_length+line_length)*j+line_length-1]
            if box.values.sum()==0:
                pseudo_df.loc[i*2,j*2]=0
            box=df.loc[(block_length+line_length)*i:(block_length+line_length)*i+line_length-1,(block_length+line_length)*j+line_length:(block_length+line_length)*j+block_length+line_length-1]
            if box.values.sum()==0:
                if j*2+1<=pseudo_df.shape[1]-1:
                    pseudo_df.loc[i*2,j*2+1]=0
    
    for i in range(0,mat_raws+1):
        for j in range(0,mat_cols+1):
            box=df.loc[(block_length+line_length)*i+line_length:(block_length+line_length)*i+line_length+block_length-1,(block_length+line_length)*j:(block_length+line_length)*j+line_length-1]
            if box.values.sum()==0:
                pseudo_df.loc[i*2+1,j*2]=0
            box=df.loc[(block_length+line_length)*i+line_length:(block_length+line_length)*i+line_length+block_length-1,(block_length+line_length)*j+line_length:(block_length+line_length)*j+block_length+line_length-1]
            if box.values.sum()==0:
                if (i*2+1<=pseudo_df.shape[0]-1)&(j*2+1<=pseudo_df.shape[1]-1):
                    pseudo_df.loc[i*2+1,j*2+1]=0
    
    pseudo_df=pseudo_df.loc[0:pseudo_df.shape[0]-2,:]
    pseudo_df.loc[0,1]=2
    pseudo_df.loc[pseudo_df.shape[0]-1,pseudo_df.shape[1]-2]=3  

    # solver
    loc_start, loc_end, initial_success_flag, searched_ones=solve_maze(pseudo_df)

    if initial_success_flag:
        data_= pd.DataFrame(np.zeros(shape=(pseudo_df.shape[0],pseudo_df.shape[1])))
        data_.loc[loc_start[0],loc_start[1]]=2
        data_.loc[loc_end[0],loc_end[1]]=3
        for item in searched_ones:
            data_.loc[item[0],item[1]]= 1

        for col in data_.columns:
            data_[col] = data_[col].astype(int)

        a_flag=True
        to_be_removed_list=[]
        searched_ones_=searched_ones.copy()
        while a_flag:
            data_= pd.DataFrame(np.zeros(shape=(pseudo_df.shape[0],pseudo_df.shape[1])))
            data_.loc[loc_start[0],loc_start[1]]=2
            data_.loc[loc_end[0],loc_end[1]]=3
            for item in searched_ones_:
                data_.loc[item[0],item[1]]= 1
            
            to_be_removed_list_temp=[]
            for item in searched_ones_:
                a_list=[get_loc_val(x)[1] for x in find_neighbour(data_,item) if get_loc_val(x)[1]!=None]
                if (sum(a_list)==1):
                    to_be_removed_list_temp.append(item)
            if len(to_be_removed_list_temp)==0:
                a_flag=False
            else:
                to_be_removed_list.extend(to_be_removed_list_temp)
                searched_ones_=[x for x in searched_ones_ if x not in to_be_removed_list_temp]

        data_= pd.DataFrame(np.zeros(shape=(pseudo_df.shape[0],pseudo_df.shape[1])))
        data_.loc[loc_start[0],loc_start[1]]=2
        data_.loc[loc_end[0],loc_end[1]]=3
        for item in searched_ones_:
            data_.loc[item[0],item[1]]= 1

        for col in data_.columns:
            data_[col] = data_[col].astype(int)

        inside_im=np.zeros([df.shape[0], df.shape[1],3], dtype=int)

        for i in range(0,df.shape[0]):
            for j in range(0,df.shape[1]):
                if df.loc[i,j]==1:
                    inside_im[i,j,:]=[255,255,255]   

        
        h=df.shape[0]/data_.shape[0]
        w=df.shape[1]/data_.shape[1]

        for i in range(0,data_.shape[0]):
            for j in range(0,data_.shape[1]):
                if data_.loc[i,j]==1:
                    inside_im[round(i*h):round((i+1)*h),round(j*w):round((j+1)*w),:]=[0,255,0]
                if data_.loc[i,j]==2:
                    inside_im[round(i*h):round((i+1)*h),round(j*w):round((j+1)*w),:]=[0,0,255]
                if data_.loc[i,j]==3:
                    inside_im[round(i*h):round((i+1)*h),round(j*w):round((j+1)*w),:]=[0,0,255]


        print("Maze has a solution")
        return inside_im

    else:
        print("Maze has no solution")
        return None