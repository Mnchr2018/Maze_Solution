from pathlib import Path
import os
import cv2
import pandas as pd
import numpy as np
import random

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

def master_solver(image, black_white_threshold, pixel_to_matrix_factor,pixel_margin, block_threshold):
    # read image and convert to a pd dataframe
    im = cv2.imread(image,2)
    df=pd.DataFrame(im)

    # convert dateframe to binary dataframe
    df=convert_to_binary(df,black_white_threshold)

    # crop image by removing white space around maze
    df=crop_df(df)

    # converting pixcel to matrix, find the number of columns
    unique_cols=[]
    for i in list(range(2, df.shape[0]-2)):
        if df.loc[i,:].sum()>=pixel_to_matrix_factor*df.shape[1]:
            unique_cols.extend([i for i, x in enumerate(df.loc[i,:].tolist()) if x==0])  
    raw_cols=sorted(list(set(unique_cols)))
    count=1
    for i,v in enumerate(raw_cols):
        if (i!=0)&(raw_cols[i-1]!=raw_cols[i]-1):
            count=count+1
    mat_cols=count-1

    # converting pixcel to matrix,find the number of rows
    unique_rows=[]
    for i in list(range(2, df.shape[1]-2)):
        if df.loc[:,i].sum()>=pixel_to_matrix_factor*df.shape[0]:
            unique_rows.extend([i for i, x in enumerate(df.loc[:,i].tolist()) if x==0])  
    raw_rows=sorted(list(set(unique_rows)))
    count=1
    for i,v in enumerate(raw_rows):
        if (i!=0)&(raw_rows[i-1]!=raw_rows[i]-1):
            count=count+1
    mat_raws=count-1

    # making a mtrix out of pixcels
    pseudo_df=pd.DataFrame(np.zeros(shape=(mat_raws*2-1,mat_cols*2-1)))
    pseudo_df=pd.DataFrame(np.where(pseudo_df==0, 1, 0))
    row_stride=df.shape[1]/(pseudo_df.shape[1]+1)
    col_stride=df.shape[0]/(pseudo_df.shape[0]+1)
    for i in range(0,pseudo_df.shape[0]):
        for j in range(0,pseudo_df.shape[1]):
            box=df.loc[range(round((i+0.5)*col_stride)+pixel_margin,round((i+1.5)*col_stride)-pixel_margin),range(round((j+0.5)*row_stride)+pixel_margin,round((j+1.5)*row_stride)-pixel_margin)]
            if box.values.sum()/(box.shape[0]*box.shape[1])>=block_threshold:
                pseudo_df.loc[i,j]=1
            else:
                pseudo_df.loc[i,j]=0  
    
    # assign 2 for starting block and 3 for end block
    pseudo_df.iloc[0,0]=2
    pseudo_df.iloc[pseudo_df.shape[0]-1,pseudo_df.shape[1]-1]=3 

    # intial solution
    loc_start, loc_end, initial_success_flag, searched_ones=solve_maze(pseudo_df)

    if initial_success_flag:
        # create a new dataframe with raw solution. no short path
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
                if (sum(a_list)==1)or((2 in a_list)&(sum(a_list)!=3)):
                    to_be_removed_list_temp.append(item)
            if len(to_be_removed_list_temp)==0:
                a_flag=False
            else:
                to_be_removed_list.extend(to_be_removed_list_temp)
                searched_ones_=[x for x in searched_ones_ if x not in to_be_removed_list_temp]
        
        # final matrix with short path
        data_= pd.DataFrame(np.zeros(shape=(pseudo_df.shape[0],pseudo_df.shape[1])))
        data_.loc[loc_start[0],loc_start[1]]=2
        data_.loc[loc_end[0],loc_end[1]]=3
        for item in searched_ones_:
            data_.loc[item[0],item[1]]= 1
        for col in data_.columns:
            data_[col] = data_[col].astype(int)
        
        # converting matrix to image
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
