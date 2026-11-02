#################################################

#    Utils in this page were attained from:
#    https://github.com/anas337/Human-Activity-Recognition-Using-Smartphones.github.io/blob/df9773951f723fa944ac05e92eefb0a03b0c45f1/signal%20processing%20pipeline.ipynb

##################################################

import numpy as np
import scipy as sp
import pandas as pd
from glob import glob
from IPython.display import display
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.fftpack import fft   # import fft(Fast Fourier Transform) function to convert a signal from time domain to 
#                               frequency domain (output :is a numpy array contains signal's amplitudes of each frequency component)

from scipy.fftpack import fftfreq # import fftfreq function to generate frequencies related to frequency components 
#                                   mentioned above
from scipy.fftpack import ifft # import ifft function (inverse fft) inverse the conversion
import math

def import_raw_signals(file_path,columns):
    ######################################################################################
    # Inputs:                                                                            #
    #   file_path: A string contains the path of the "acc" or "gyro" txt file            #
    #   columns: A list of strings contains the column names in order.                   #
    # Outputs:                                                                           #
    #   dataframe: A pandas Dataframe contains "acc" or "gyro" data in a float format    #
    #             with columns names.                                                    #
    ######################################################################################


    # open the txt file
    opened_file =open(file_path,'r')

    # Create a list
    opened_file_list=[]
    
    # loop over each line in the opened_file
    # convert each element from txt format to float 
    # store each raw in a list
    for line in opened_file:
        opened_file_list.append([float(element) for element in line.split()])

    # convert the list of lists into 2D numpy array(computationally efficient)
    data=np.array(opened_file_list)


    # Create a pandas dataframe from this 2D numpy array with column names
    data_frame=pd.DataFrame(data=data,columns=columns)

    # return the data frame
    return data_frame

def import_labels_file(path,columns):
    ######################################################################################
    # Inputs:                                                                            #
    #   path: A string contains the path of "labels.txt"                                 #
    #   columns: A list of strings contains the columns names in order.                  #
    # Outputs:                                                                           #
    #   dataframe: A pandas Dataframe contains labels  data in int format                #
    #             with columns names.                                                    #
    ######################################################################################
    
    
    # open the txt file
    labels_file =open(path,'r')
    
    # creating a list 
    labels_file_list=[]
    
    
    #Store each row in a list ,convert its list elements to int type
    for line in labels_file:
        labels_file_list.append([int(element) for element in line.split()])
    # convert the list of lists into 2D numpy array 
    data=np.array(labels_file_list)
    
    # Create a pandas dataframe from this 2D numpy array with column names 
    data_frame=pd.DataFrame(data=data,columns=columns)
    
    # returning the labels dataframe 
    return data_frame


sampling_freq=50 # 50 Hz(hertz) is sampling frequency: the number of captured values of each axial signal per second.



def visualize_triaxial_signals(data_frame,exp_id,act,sig_type,width,height, raw_dic):
    
    #################################### INPUTS ####################################################################
    # inputs: Data_frame: Data frame contains acc and gyro signals                                                 #
    #         exp_id: integer from 1 to 61 (the experience identifier)                                             #         
    #         width: integer the width of the figure                                                               #
    #         height: integer the height of the figure                                                             #
    #         sig_type: string  'acc' to visualize 3-axial acceleration signals or 'gyro' for 3-axial gyro signals #
    #         act: possible values: string: 'all' (to visualize full signals) ,                                    #
    #              or integer from 1 to 12 to specify the activity id to be visualized                             #
    #                                                                                                              #
    #              if act is from 1 to 6 it will skip the first 250 rows(first 5 seconds) from                     #
    #              the starting point of the activity and will visualize the next 400 rows (next 8 seconds)        #
    #              if act is between 7 and 12  the function will visualize all rows(full duration) of the activity.# 
    #################################################################################################################
    
    
    keys=sorted(raw_dic.keys()) # list contains 'expXX_userYY' sorted from 1 to 61
    key=keys[exp_id-1] # the key associated to exp_id (experience)
    exp_id=str(exp_id)
    user_id=key[-2:] # the user id associated to this experience in string format
    
    if act=='all': # to visualize full signal
        # selecting all rows in the dataframe to be visualized , the dataframe stored in raw_dic and has the same key
        data_df=data_frame
    else:# act is an integer from 1 to 12 (id of the activity to be visualized ) 
        # Select rows in labels file having the same exp_Id and user_Id mentioned above + the activity id (act)
        # selecting the first result in the search made in labels file
        # and select the start point and end point of this row related to this activity Id (act) 
        start_point,end_point=Labels_Data_Frame[
                             (Labels_Data_Frame["experiment_number_ID"]==int(exp_id))&
                             (Labels_Data_Frame["user_number_ID"]==int(user_id))&
                             (Labels_Data_Frame["activity_number_ID"]==act)
                            
                            ][['Label_start_point','Label_end_point']].iloc[0]
        
        if int(act) in [1,2,3,4,5,6]:# if the activity to be visualed is from 1 to 6 (basic activity)
            # skip the first 250 rows(5 second)
            start_point=start_point+250
            
            # set the end point at distance of 400 rows (8seconds) from the start_point
            end_point=start_point+400
    
        
        # select the 8 seconds shifted by 5 seconds if act between 1 and 6 
        # if act is between 7 and 12 select the full duration of the first result(row)
        data_df=data_frame[start_point:end_point]
    
    ##################################
    
    columns=data_df.columns # a list contain all column names of the  (6 columns in total)

    if sig_type=='acc':# if the columns to be visualized are acceleration columns
        
        # acceleration columns are the first 3 columns acc_X, acc_Y and acc_Z
        X_component=data_df[columns[0]] # copy acc_X
        Y_component=data_df[columns[1]] # copy acc_Y
        Z_component=data_df[columns[2]] # copy acc_Z
        
        # accelerations legends
        legend_X='acc_X'
        legend_Y='acc_Y'
        legend_Z='acc_Z'

        # the figure y axis info
        figure_Ylabel='Acceleration in 1g'
        
        # select the right title in each case
        
        if act=='all':
            title="acceleration signals for all activities performed by user "+ user_id +' in experience '+exp_id
        
        elif act in [1,2,3,4,5,6,7,8,9,10,11,12]:
            title="acceleration signals of experience "+ exp_id+" while user "+ user_id +' was performing activity: '+str(act)+ '('+AL[act]+')'

    elif sig_type=='gyro':# if the columns to be visualized are gyro columns
        
        # gyro columns are the last 3 columns gyro_X, gyro_Y and gyro_Z
        X_component=data_df[columns[3]] # copy gyro_X
        Y_component=data_df[columns[4]] # copy gyro_Y
        Z_component=data_df[columns[5]] # copy gyro_Z
        
        # gyro signals legends
        legend_X='gyro_X'
        legend_Y='gyro_Y'
        legend_Z='gyro_Z'
        
        #the figure y axis info
        figure_Ylabel='Angular Velocity in radian per second [rad/s]'
        
        # select the right title in each case 
        if act=='all':
            title="gyroscope signals for all activities performed by user "+ user_id +' in experience '+exp_id
        elif act in [1,2,3,4,5,6,7,8,9,10,11,12]:
            title="gyroscope signals of experience "+ exp_id+" while user "+ user_id +' was performing activity: '+str(act)+ '('+AL[act]+')'

    # chosing colors : red for X component blue for Y component and green for Z component
    colors=['r','b','g']
    len_df=len(data_df) # number of rows in this dataframe to be visualized(depends on 'act' variable)
    
    # converting row numbers into time duration (the duration between two rows is 1/50=0.02 second)
    time=[1/float(sampling_freq) *j for j in range(len_df)]
    
    # Define the figure and setting dimensions width and height
    fig = plt.figure(figsize=(width,height))
    
    # ploting each signal component
    _ =plt.plot(time,X_component,color='r',label=legend_X)
    _ =plt.plot(time,Y_component,color='b',label=legend_Y)
    _ =plt.plot(time,Z_component,color='g',label=legend_Z)
    
    # Set the figure info defined earlier
    _ = plt.ylabel(figure_Ylabel) # set Y axis info 
    _ = plt.xlabel('Time in seconds (s)') # Set X axis info (same label in all cases)
    _ = plt.title(title) # Set the title of the figure
    
    # localise the figure's legends
    _ = plt.legend(loc="upper left")# upper left corner
    
    # showing the figure
    plt.show()

def visualize_signal(signal,x_labels,y_labels,title,legend):
    # Inputs: signal: 1D column 
    #         x_labels: the X axis info (figure)
    #         y_labels: the Y axis info (figure)
    #         title: figure's title
    #         legend : figure's legend
    
    
    
    # Define the figure's dimensions
    plt.figure(figsize=(20,4))
    
    # convert row numbers in time durations
    time=[1/float(sampling_freq) *i for i in range(len(signal))]
    
    # plotting the signal
    plt.plot(time,signal,label=legend) # plot the signal and add the legend
    
    plt.xlabel(x_labels) # set the label of x axis in the figure
    plt.ylabel(y_labels) # set the label of y axis in the figure
    plt.title(title) # set the title of the figure
    plt.legend(loc="upper left") # set the legend in the upper left corner
    plt.show() # show the figure


def components_selection_one_signal(t_signal,freq1,freq2):
    t_signal=np.array(t_signal)
    t_signal_length=len(t_signal) # number of points in a t_signal
    
    # the t_signal in frequency domain after applying fft
    f_signal=fft(t_signal) # 1D numpy array contains complex values (in C)
    
    # generate frequencies associated to f_signal complex values
    freqs=np.array(sp.fftpack.fftfreq(t_signal_length, d=1/float(sampling_freq))) # frequency values between [-25hz:+25hz]
    
    # DC_component: f_signal values having freq between [-0.3 hz to 0 hz] and from [0 hz to 0.3hz] 
    #                                                             (-0.3 and 0.3 are included)
    
    # noise components: f_signal values having freq between [-25 hz to 20 hz[ and from ] 20 hz to 25 hz] 
    #                                                               (-25 and 25 hz inculded 20hz and -20hz not included)
    
    # selecting body_component: f_signal values having freq between [-20 hz to -0.3 hz] and from [0.3 hz to 20 hz] 
    #                                                               (-0.3 and 0.3 not included , -20hz and 20 hz included)
    
    
    f_DC_signal=[] # DC_component in freq domain
    f_body_signal=[] # body component in freq domain numpy.append(a, a[0])
    f_noise_signal=[] # noise in freq domain
    
    for i in range(len(freqs)):# iterate over all available frequencies
        
        # selecting the frequency value
        freq=freqs[i]
        
        # selecting the f_signal value associated to freq
        value= f_signal[i]
        
        # Selecting DC_component values 
        if abs(freq)>0.3:# testing if freq is outside DC_component frequency ranges
            f_DC_signal.append(float(0)) # add 0 to  the  list if it was the case (the value should not be added)                                       
        else: # if freq is inside DC_component frequency ranges 
            f_DC_signal.append(value) # add f_signal value to f_DC_signal list
    
        # Selecting noise component values 
        if (abs(freq)<=20):# testing if freq is outside noise frequency ranges 
            f_noise_signal.append(float(0)) # # add 0 to  f_noise_signal list if it was the case 
        else:# if freq is inside noise frequency ranges 
            f_noise_signal.append(value) # add f_signal value to f_noise_signal

        # Selecting body_component values 
        if (abs(freq)<=0.3 or abs(freq)>20):# testing if freq is outside Body_component frequency ranges
            f_body_signal.append(float(0))# add 0 to  f_body_signal list
        else:# if freq is inside Body_component frequency ranges
            f_body_signal.append(value) # add f_signal value to f_body_signal list
    
    ################### Inverse the transformation of signals in freq domain ########################
    # applying the inverse fft(ifft) to signals in freq domain and put them in float format
    t_DC_component= ifft(np.array(f_DC_signal)).real
    t_body_component= ifft(np.array(f_body_signal)).real
    t_noise=ifft(np.array(f_noise_signal)).real
    
    total_component=t_signal-t_noise # extracting the total component(filtered from noise) 
                                     #  by substracting noise from t_signal (the original signal).
    
    # return outputs mentioned earlier
    return (total_component,t_DC_component,t_body_component,t_noise) 



def mag_3_signals(x,y,z):# magnitude function redefintion
    return np.array([math.sqrt((x[i]**2+y[i]**2+z[i]**2)) for i in range(len(x))])

def verify_gravity(exp_id, raw_dic, freq1, freq2):
    # inputs: exp_id integer from 1 to 61
    
    keys=sorted(raw_dic.keys())# expXX_userYY ordered from 1 to 61
    key=keys[exp_id-1]# select the key associated to exp_id
    
    acc_x=np.array(raw_dic[key]['acc_X']) # copy acc_X column from dataframe in raw_dic having the key mentioned above
    acc_y=np.array(raw_dic[key]['acc_Y'])# copy acc_Y column  from dataframe in raw_dic having the key mentioned above
    acc_z=np.array(raw_dic[key]['acc_Z'])# copy acc_Z column  from dataframe in raw_dic having the key mentioned above

    # apply the filtering method to acc_[X,Y,Z] and store gravity components
    grav_acc_X=components_selection_one_signal(acc_x,freq1,freq2)[1] 
    grav_acc_Y=components_selection_one_signal(acc_y,freq1,freq2)[1]
    grav_acc_Z=components_selection_one_signal(acc_z,freq1,freq2)[1]
    
    # calculating gravity magnitude signal
    grav_acc_mag=mag_3_signals(grav_acc_X, grav_acc_Y,grav_acc_Z)

    x_labels='time in seconds' # Set x axis info
    y_labels='gravity amplitude in 1g' # set y axis info
    title='the euclidian magnitude of gravity 3-axial signals' # set the figure's title
    legend=key+' grav_acc_mag' # set the figure's legend
    
    visualize_signal(grav_acc_mag,x_labels,y_labels,title,legend) # visualize gravity magnitude signal
    print('mean value = '+str(np.array(grav_acc_mag).mean())[0:5]+ ' g') # print the gravity magnitude mean value



dt=0.02 # dt=1/50=0.02s time duration between two rows
# Input: 1D array with lenght=N (N:unknown)
# Output: 1D array with lenght=N-1
def jerk_one_signal(signal): 
        return np.array([(signal[i+1]-signal[i])/dt for i in range(len(signal)-1)])


def median(signal):# input: numpy array 1D (one column)
    array=np.array(signal)   
    #applying the median filter
    med_filtered=sp.signal.medfilt(array, kernel_size=3) # applying the median filter order3(kernel_size=3)
    return  med_filtered # return the med-filtered signal: numpy array 1D

# example: 679 ==> '00679'; 50 ==> '00050'

# it add '0's to the left of the input until the new lenght is equal to 5
def normalize5(number): 
    stre=str(number)
    if len(stre)<5:
        l=len(stre)
        for i in range(0,5-l):
            stre="0"+stre
    return stre 

# it add '0's to the left of the input until the new lenght is equal to 2
def normalize2(number):
    stre=str(number)
    if len(stre)<2:
        stre="0"+stre
    return stre

def Windowing_type_1(time_sig_dic,Labels_Data_Frame):   
    
    columns=time_sig_dic['exp01_user01'].columns # columns of time_sig_df
    window_ID=0 # window unique id
    t_dic_win_type_I={} # output dic
    
    BA_array=np.array(Labels_Data_Frame[(Labels_Data_Frame["activity_number_ID"] <7)]) # Just Basic activities
    
    for line in BA_array:
        # Each line in BA_array contains info realted to an activity

        # extracting the dataframe key that contains rows related to this activity [expID,userID]
        file_key= 'exp' + normalize2(int(line[0]))  +  '_user' + normalize2(int(line[1]))

        # extract the activity id in this line
        act_ID=line[2] # The activity identifier from 1 to 6 (6 included)

        # starting point index of an activity
        start_point=line[3]

        # from the cursor we copy a window that has 128 rows
        # the cursor step is 64 data point (50% of overlap) : each time it will be shifted by 64 rows
        for cursor in range(start_point,line[4]-127,64):

            # end_point: cursor(the first index in the window) + 128
            end_point=cursor+128 # window end row

            # selecting window data points convert them to numpy array to delete rows index
            data=np.array(time_sig_dic[file_key].iloc[cursor:end_point])

            # converting numpy array to a dataframe with the same column names
            window=pd.DataFrame(data=data,columns=columns)

            # creating the window
            key='t_W'+normalize5(window_ID)+'_'+file_key+'_act'+normalize2(act_ID)
            t_dic_win_type_I[key]=window

            # incrementing the windowID by 1
            window_ID=window_ID+1
        
    return t_dic_win_type_I # return a dictionary including time domain windows type I

def find_activity(Labels_Data_Frame,exp_ID,cursor,end_point):
    
    # array contains all rows from Labels_Data_Frame having related to experience :exp_ID
    exp_AL=np.array(Labels_Data_Frame[Labels_Data_Frame['experiment_number_ID']==exp_ID])
    
    # the near starting point above(inferior) the cursor
    St_p1=exp_AL[exp_AL[:,3]<=cursor][:,3].max()  # starting point label of the first activity
    
    # the near ending point below the cursor (superior ) 
    En_p2=exp_AL[exp_AL[:,4]>=end_point][:,4].min() # ending point label of the second activity

    for index in range(len(exp_AL)): # iterating over each line 
        
        if exp_AL[index,3]==St_p1: # selecting the row index with starting point 
            Ind1=index # index of the first activity
        
        if exp_AL[index,4]==En_p2: # selecting the row index with ending point ==b
            Ind2=index # the index of the second activity
    
    if Ind1 == Ind2 : # if the window rows indexes are inside the same activity data points which means 
        #                that the first and second activity are actually the same 
        
        activity=exp_AL[Ind1,2] # window_activity id will take the same value as the activity_label of these rows
    
    else:
        
        # exp_AL[Ind1,4] is the ending_point_label of the first activity
        if cursor+63<= exp_AL[Ind1,4]:# if the first 64 data points or more  are included in the first activity bands
            
            activity=exp_AL[Ind1,2] # the window will take the same activity labels as the first activity
        # exp_AL[Ind2,3] is the starting_point_label of the second activity
        
        elif cursor+64>= exp_AL[Ind2,3]:# if the last 64 data points or more  are included in the second activity bands
            activity=exp_AL[Ind2,2] # the window will take the activity labels as the second activity
        
        else: # if  more than 64 data point doesn't belong (all at once) neither to first activity or to the second activity  
            activity=None # this window activity label will be equal to None
    
    if activity!=None:
        return normalize2(activity) # to convert to string and add a '0' to the left if activity <10
    else:
        return None

def Windowing_type_2(time_sig_dic,Labels_Data_Frame):
    columns=time_sig_dic['exp01_user01'].columns
    window_ID=0
    t_dic_win_type_II={}

    for key in sorted(time_sig_dic.keys()):

        exp_array=np.array(time_sig_dic[key]) # converting the data frame in a an array
        lenght_exp=len(exp_array) # number of rows in this array

        # Extracting the experience number ID an convert it to int 
        exp_ID=int(key[3:5])
        #user_ID=int(key[10:])

        # Selecting Labels_Data_Frame rows blonging to the same experiment ID : exp_ID
        exp_AL_array=np.array( Labels_Data_Frame[ Labels_Data_Frame['experiment_number_ID'] == exp_ID])

        # The first starting_point_label in this experiment
        start_point=exp_AL_array[0,3]

        # The last ending_point_label of this experiment
        end_point=exp_AL_array[-1,4]

        for cursor in range(start_point,end_point-127,64): # the cursor always represents the index of first data point in a window
            end_window=cursor+127 # end_window is the index of the last data point in a window

            # creating the window
            window_array=exp_array[cursor:end_window+1,:]

            # Determining the appropriate activity label of this window
            act_ID=find_activity(Labels_Data_Frame,exp_ID,cursor,end_window) # strings from 01 to 12 or None

            if act_ID!=None: # if act_ID is none  this window doesn't belong to any activity

                # since the act_ID is != to None the window array will be stored in DataFrame with the appropriate column names
                window_df=pd.DataFrame(data=window_array,columns=columns)

                # Generating the window key(unique key :since the window_ID is unique for each window )
                # I choosed to add the exp, user, activity Identifiers in the win_key they will be usefull later. (exp: optional)

                win_key='t_W'+normalize5(window_ID)+'_'+key+'_act'+str(act_ID)# eg: 'W_00000_exp01_user01_act01'

                # Store the window data frame in a dic
                t_dic_win_type_II[win_key]=window_df

                # Incrementing window_ID by 1
                window_ID=window_ID+1
    
    return t_dic_win_type_II

from scipy import fftpack # import fftpack to use all fft functions
from numpy.fft import *

##################### fast_fourier_transform_one_signal #################
# Inputs: time signal 1D array
# Output: amplitude of fft components 1D array having the same lenght as the Input
def fast_fourier_transform_one_signal(t_signal):
    # apply fast fourrier transform to the t_signal
    complex_f_signal= fftpack.fft(t_signal)
    #compute the amplitude each complex number
    amplitude_f_signal=np.abs(complex_f_signal)
    # return the amplitude
    return amplitude_f_signal

##################### fast fourier transform for data frames #################
# Inputs: A DataFrame with 20 time signal (20 columns) gravity columns(4) won't be transformed
# Outputs: A DataFrame with 16 frequency signal (16 columns)
def fast_fourier_transform(t_window):
    
    f_window=pd.DataFrame() # create an empty dataframe will include frequency domain signals of window
    
    for column in t_window.columns: # iterating over time domain window columns(signals)
        
        if 'grav' not in column: # verify if time domain signal is not related to gravity components
            
            t_signal=np.array(t_window[column]) # convert the column to a 1D numpy array
           
            f_signal= np.apply_along_axis(fast_fourier_transform_one_signal,0,t_signal) # apply the function defined above to the column
            f_window["f_"+column[2:]]=f_signal # storing the frequency signal in f_window with an appropriate column name
    
    return f_window # return the frequency domain window


# mean
def mean_axial(df):
    array=np.array(df) # convert dataframe into 2D numpy array for efficiency
    mean_vector = list(array.mean(axis=0)) # calculate the mean value of each column
    return mean_vector # return mean vetor
# std
def std_axial(df):
    array=np.array(df)# convert dataframe into 2D numpy array for efficiency
    std_vector = list(array.std(axis=0))# calculate the standard deviation value of each column
    return std_vector

# mad
from statsmodels.robust import mad as median_deviation # import the median deviation function
def mad_axial(df):
    array=np.array(df)# convert dataframe into 2D numpy array for efficiency
    mad_vector = list(median_deviation(array,axis=0)) # calculate the median deviation value of each column
    return mad_vector

# max

def max_axial(df):
    array=np.array(df)# convert dataframe into 2D numpy array for efficiency
    max_vector=list(array.max(axis=0))# calculate the max value of each column
    return max_vector
# min
def min_axial(df):
    array=np.array(df)# convert dataframe into 2D numpy array for efficiency
    min_vector=list(array.min(axis=0))# calculate the min value of each column
    return min_vector
# IQR
from scipy.stats import iqr as IQR # import interquartile range function (Q3(column)-Q1(column))
def IQR_axial(df):
    array=np.array(df)# convert dataframe into 2D numpy array for efficiency
    IQR_vector=list(np.apply_along_axis(IQR,0,array))# calculate the inter quartile range value of each column
    return IQR_vector


# Entropy
from scipy.stats import entropy # import the entropy function
def entropy_axial(df):
    array=np.array(df)# convert dataframe into 2D numpy array for efficiency
    entropy_vector=list(np.apply_along_axis(entropy,0,abs(array)))# calculate the entropy value of each column
    return entropy_vector

def mean_mag(mag_column):
    array=np.array(mag_column)
    mean_value = float(array.mean())
    return mean_value

# std: standard deviation of mag column
def std_mag(mag_column):
    array=np.array(mag_column)
    std_value = float(array.std()) # std value 
    return std_value

# mad: median deviation
def mad_mag(mag_column):
    array=np.array(mag_column)
    mad_value = float(median_deviation(array))# median deviation value of mag_column
    return mad_value

# max
def max_mag(mag_column):
    array=np.array(mag_column)
    max_value=float(array.max()) # max value 
    return max_value
# min
def min_mag(mag_column):
    array=np.array(mag_column)
    min_value= float(array.min()) # min value
    return min_value

# IQR
def IQR_mag(mag_column):
    array=np.array(mag_column)
    IQR_value=float(IQR(array))# Q3(column)-Q1(column)
    return IQR_value

# Entropy
def entropy_mag(mag_column):
    array=np.array(mag_column)
    entropy_value=float(entropy(array)) # entropy signal
    return entropy_value

def t_sma_axial(df):
    array=np.array(df)
    sma_axial=float(abs(array).sum())/float(3) # sum of areas under each signal
    return sma_axial # return sma value

# energy
def t_energy_axial(df):
    array=np.array(df)
    energy_vector=list((array**2).sum(axis=0)) # energy value of each df column
    return energy_vector # return energy vector energy_X,energy_Y,energy_Z

# AR vector (auto regression coefficients from 1 to 4)

# define the arbugr function
#auto regression coefficients with using burg method with order from 1 to 4
from spectrum import *

##############################################################################################
# I took this function as it is from this link ------>    https://github.com/faroit/freezefx/blob/master/fastburg.py
# This fucntion and the original function arburg in the library spectrum generate the same first 3 coefficients 
#for all windows the original burg method is low and for some windows it cannot generate all 4th coefficients 

def _arburg2(X, order):
    """This version is 10 times faster than arburg, but the output rho is not correct.
    returns [1 a0,a1, an-1]
    """
    x = np.array(X)
    N = len(x)

    if order == 0.:
        raise ValueError("order must be > 0")

    # Initialisation
    # ------ rho, den
    rho = sum(abs(x)**2.) / N  # Eq 8.21 [Marple]_
    den = rho * 2. * N

    # ------ backward and forward errors
    ef = np.zeros(N, dtype=complex)
    eb = np.zeros(N, dtype=complex)
    for j in range(0, N):  # eq 8.11
        ef[j] = x[j]
        eb[j] = x[j]

    # AR order to be stored
    a = np.zeros(1, dtype=complex)
    a[0] = 1
    # ---- rflection coeff to be stored
    ref = np.zeros(order, dtype=complex)

    E = np.zeros(order+1)
    E[0] = rho

    for m in range(0, order):
        # print m
        # Calculate the next order reflection (parcor) coefficient
        efp = ef[1:]
        ebp = eb[0:-1]
        # print efp, ebp
        num = -2. * np.dot(ebp.conj().transpose(), efp)
        den = np.dot(efp.conj().transpose(),  efp)
        den += np.dot(ebp,  ebp.conj().transpose())
        ref[m] = num / den

        # Update the forward and backward prediction errors
        ef = efp + ref[m] * ebp
        eb = ebp + ref[m].conj().transpose() * efp

        # Update the AR coeff.
        a.resize(len(a)+1)
        a = a + ref[m] * np.flipud(a).conjugate()

        # Update the prediction error
        E[m+1] = np.real((1 - ref[m].conj().transpose() * ref[m])) * E[m]
        # print 'REF', ref, num, den
    return a, E[-1], ref

#################################################################################################################

# to generate arburg (order 4) coefficents for 3 columns [X,Y,Z]
def t_arburg_axial(df):
    # converting signals to 1D numpy arrays for efficiency
    array_X=np.array(df[df.columns[0]])
    array_Y=np.array(df[df.columns[1]])
    array_Z=np.array(df[df.columns[2]])
    
    AR_X = list(_arburg2(array_X,4)[0][1:].real) # list contains real parts of all 4th coefficients generated from signal_X
    AR_Y = list(_arburg2(array_Y,4)[0][1:].real) # list contains real parts of all 4th coefficients generated from signal_Y
    AR_Z = list(_arburg2(array_Z,4)[0][1:].real) # list contains real parts of all 4th coefficients generated from signal_Z
    
    # selecting [AR1 AR2 AR3 AR4] real components for each axis concatenate them in one vector
    AR_vector= AR_X + AR_Y+ AR_Z
    
    
    # AR_vector contains 12 values 4values per each axis 
    return AR_vector


from scipy.stats import pearsonr
def t_corr_axial(df): # it returns 3 correlation features per each 3-axial signals in  time_window
    
    array=np.array(df)
    
    Corr_X_Y=float(pearsonr(array[:,0],array[:,1])[0]) # correlation value between signal_X and signal_Y
    Corr_X_Z=float(pearsonr(array[:,0],array[:,2])[0]) # correlation value between signal_X and signal_Z
    Corr_Y_Z=float(pearsonr(array[:,1],array[:,2])[0]) # correlation value between signal_Y and signal_Z
    
    corr_vector =[Corr_X_Y, Corr_X_Z, Corr_Y_Z] # put correlation values in list
    
    return corr_vector 

def t_axial_features_generation(t_window):
    
    # select axial columns : the first 15 columns
    axial_columns=t_window.columns[0:15]
    
    # select axial columns in a dataframe
    axial_df=t_window[axial_columns]
    
    ## a list will contain all axial features values resulted from applying: 
    #  common axial features functions and time axial features functions to all time domain signals in t_window
    t_axial_features=[]
    for col in range(0,15,3):
        df=axial_df[axial_columns[col:col+3]] # select each group of 3-axial signal: signal_name[X,Y,Z]
        
        # apply all common axial features functions and time axial features functions to each 3-axial signals dataframe
        mean_vector   = mean_axial(df) # 3values
        std_vector    = std_axial(df) # 3 values
        mad_vector    = mad_axial(df)# 3 values
        max_vector    = max_axial(df)# 3 values
        min_vector    = min_axial(df)# 3 values
        sma_value     = t_sma_axial(df)# 1 value
        energy_vector = t_energy_axial(df)# 3 values
        IQR_vector    = IQR_axial(df)# 3 values
        entropy_vector= entropy_axial(df)# 3 values
        AR_vector     = t_arburg_axial(df)# 3 values
        corr_vector   = t_corr_axial(df)# 3 values
        
        # 40 value per each 3-axial signals
        t_3axial_vector= mean_vector + std_vector + mad_vector + max_vector + min_vector + [sma_value] + energy_vector + IQR_vector + entropy_vector + AR_vector + corr_vector
        
        # append these features to the global list of features
        t_axial_features= t_axial_features+ t_3axial_vector
    
    # t_axial_features contains 200 values = 40 value per each 3axial x 5 tri-axial-signals[X,Y,Z]
    return t_axial_features


# sma: signal magnitude area
def t_sma_mag(mag_column):
    array=np.array(mag_column)
    sma_mag=float(abs(array).sum())# signal magnitude area of one mag column
    return sma_mag

# energy
def t_energy_mag(mag_column):
    array=np.array(mag_column)
    energy_value=float((array**2).sum()) # energy of the mag signal
    return energy_value



# arburg: auto regression coefficients using the burg method
def t_arburg_mag(mag_column):
    
    array = np.array(mag_column)
    
    AR_vector= list(_arburg2(array,4)[0][1:].real) # AR1, AR2, AR3, AR4 of the mag column
    #print(AR_vector)
    return AR_vector

def t_mag_features_generation(t_window):
    
    # select mag columns : the last 5 columns in a time domain window
    
    mag_columns=t_window.columns[15:] # mag columns' names
    mag_columns=t_window[mag_columns] # mag data frame
    
    t_mag_features=[] # a global list will contain all time domain magnitude features
    
    for col in mag_columns: # iterate throw each mag column
        
        mean_value   = mean_mag(mag_columns[col]) # 1 value
        std_value    = std_mag(mag_columns[col])# 1 value
        mad_value    = mad_mag(mag_columns[col])# 1 value
        max_value    = max_mag(mag_columns[col])# 1 value
        min_value    = min_mag(mag_columns[col])# 1 value
        sma_value    = t_sma_mag(mag_columns[col])# 1 value
        energy_value = t_energy_mag(mag_columns[col])# 1 value
        IQR_value    = IQR_mag(mag_columns[col])# 1 value
        entropy_value= entropy_mag(mag_columns[col])# 1 value
        AR_vector    = t_arburg_mag(mag_columns[col])# 1 value
        
        # 13 value per each t_mag_column
        col_mag_values = [mean_value, std_value, mad_value, max_value, min_value, sma_value, 
                          energy_value,IQR_value, entropy_value]+ AR_vector
        
        # col_mag_values will be added to the global list
        t_mag_features= t_mag_features+ col_mag_values
    
    # t_mag_features contains 65 values = 13 values (per each t_mag_column) x 5 (t_mag_columns)
    return t_mag_features


def time_features_names():
    # Generating time feature names
    
    # time domain axial signals' names
    t_axis_signals=[['t_body_acc_X','t_body_acc_Y','t_body_acc_Z'],
                    ['t_grav_acc_X','t_grav_acc_Y','t_grav_acc_Z'],
                    ['t_body_acc_jerk_X','t_body_acc_jerk_Y','t_body_acc_jerk_Z'],    

                    ['t_body_gyro_X','t_body_gyro_Y','t_body_gyro_Z'],
            ['t_body_gyro_Jerk_X','t_body_gyro_Jerk_Y','t_body_gyro_Jerk_Z'],]
    
    # time domain magnitude signals' names
    magnitude_signals=['t_body_acc_Mag','t_grav_acc_Mag','t_body_acc_jerk_Mag','t_body_gyro_Mag','t_body_gyro_Jerk_Mag']

    # functions' names:
    t_one_input_features_name1=['_mean()','_std()','_mad()','_max()','_min()']

    t_one_input_features_name2=['_energy()','_iqr()','_entropy()']

    t_one_input_features_name3=['_AR1()','_AR2()','_AR3()','_AR4()']

    correlation_columns=['_Corr(X,Y)','_Corr(X,Z)','_Corr(Y,Z)']

    

    features=[]# Empty list : it will contain all time domain features' names
    
    for columns in t_axis_signals: # iterate throw  each group of 3-axial signals'
        
        for feature in t_one_input_features_name1: # iterate throw the first list of functions names
            
            for column in columns: # iterate throw each axial signal in that group
                
                newcolumn=column[:-2]+feature+column[-2:] # build the feature name
                features.append(newcolumn) # add it to the global list
        
        sma_column=column[:-2]+'_sma()' # build the feature name sma related to that group
        features.append(sma_column) # add the feature to the list
        
        for feature in t_one_input_features_name2: # same process for the second list of features functions
            for column in columns:
                newcolumn=column[:-2]+feature+column[-2:]
                features.append(newcolumn)
        
        for column in columns:# same process for the third list of features functions
            for feature in t_one_input_features_name3:
                newcolumn=column[0:-2]+feature+column[-2:]
                features.append(newcolumn)
        
        for feature in correlation_columns: # adding correlations features
            newcolumn=column[0:-2]+feature
            features.append(newcolumn)

    for columns in magnitude_signals: # iterate throw time domain magnitude column names

        # build feature names related to that column
        #list 1
        for feature in t_one_input_features_name1:
            newcolumn=columns+feature
            features.append(newcolumn)
        # sma feature name
        sma_column=columns+'_sma()'
        features.append(sma_column)
        
        # list 2
        for feature in t_one_input_features_name2: 
            newcolumn=columns+feature
            features.append(newcolumn)
        
        # list 3
        for feature in t_one_input_features_name3:
            newcolumn=columns+feature
            features.append(newcolumn)
    ###########################################################################################################
    time_list_features=features
    
    return time_list_features # return all time domain features' names

# Functions used to generate frequency axial features
# each df here is dataframe contains 3 columns (3 axial frequency domain signals X,Y,Z)
# signals were obtained from frequency domain windows
# sma
def f_sma_axial(df):
    
    array=np.array(df)
    sma_value=float((abs(array)/math.sqrt(128)).sum())/float(3) # sma value of 3-axial f_signals
    
    return sma_value

# energy
def f_energy_axial(df):
    
    array=np.array(df)
    
    # spectral energy vector
    energy_vector=list((array**2).sum(axis=0)/float(len(array))) # energy of: f_signalX,f_signalY, f_signalZ
    
    return energy_vector # enrgy veactor=[energy(signal_X),energy(signal_Y),energy(signal_Z)]


####### Max Inds and Mean_Freq Functions#######################################
# built frequencies list (each column contain 128 value)
# duration between each two successive captures is 0.02 s= 1/50hz
freqs=sp.fftpack.fftfreq(128, d=0.02) 
                                

# max_Inds
def f_max_Inds_axial(df):
    array=np.array(df)
    max_Inds_X =freqs[array[1:65,0].argmax()+1] # return the frequency related to max value of f_signal X
    max_Inds_Y =freqs[array[1:65,1].argmax()+1] # return the frequency related to max value of f_signal Y
    max_Inds_Z =freqs[array[1:65,2].argmax()+1] # return the frequency related to max value of f_signal Z
    max_Inds_vector= [max_Inds_X,max_Inds_Y,max_Inds_Z]# put those frequencies in a list
    return max_Inds_vector

# mean freq()
def f_mean_Freq_axial(df):
    array=np.array(df)
    
    # sum of( freq_i * f_signal[i])/ sum of signal[i]
    mean_freq_X = np.dot(freqs,array[:,0]).sum() / float(array[:,0].sum()) #  frequencies weighted sum using f_signalX
    mean_freq_Y = np.dot(freqs,array[:,1]).sum() / float(array[:,1].sum()) #  frequencies weighted sum using f_signalY 
    mean_freq_Z = np.dot(freqs,array[:,2]).sum() / float(array[:,2].sum()) #  frequencies weighted sum using f_signalZ
    
    mean_freq_vector=[mean_freq_X,mean_freq_Y,mean_freq_Z] # vector contain mean frequencies[X,Y,Z]
    
    return  mean_freq_vector

###################################################################################

########## Skewness & Kurtosis Functions #######################################
from scipy.stats import kurtosis       # kurtosis function
from scipy.stats import skew           # skewness function
    
def f_skewness_and_kurtosis_axial(df):
    array=np.array(df)
    
    skew_X= skew(array[:,0])  # skewness value of signal X
    kur_X= kurtosis(array[:,0])  # kurtosis value of signal X
    
    skew_Y= skew(array[:,1]) # skewness value of signal Y
    kur_Y= kurtosis(array[:,1])# kurtosis value of signal Y
    
    skew_Z= skew(array[:,2])# skewness value of signal Z
    kur_Z= kurtosis(array[:,2])# kurtosis value of signal Z
    
    skew_kur_3axial_vector=[skew_X,kur_X,skew_Y,kur_Y,skew_Z,kur_Z] # return the list
    
    return skew_kur_3axial_vector
##################################################################################


#################### Bands Energy FUNCTIONS ########################

# bands energy levels (start row,end_row) end row not included 
B1=[(1,9),(9,17),(17,25),(25,33),(33,41),(41,49),(49,57),(57,65)] 
B2=[(1,17),(17,31),(31,49),(49,65)]
B3=[(1,25),(25,49)]

def f_one_band_energy(f_signal,band): # f_signal is one column in frequency axial signals in f_window
    # band: is one tuple in B1 ,B2 or B3 
    f_signal_bounded = f_signal[band[0]:band[1]] # select f_signal components included in the band
    energy_value=float((f_signal_bounded**2).sum()/float(len(f_signal_bounded))) # energy value of that band
    return energy_value

def f_all_bands_energy_axial(df): # df is dataframe contain 3 columns (3-axial f_signals [X,Y,Z])
    
    E_3_axis =[]
    
    array=np.array(df)
    for i in range(0,3): # iterate throw signals
        E1=[ f_one_band_energy( array,( B1 [j][0], B1 [j][1]) ) for j in range(len(B1))] # energy bands1 values of f_signal
        E2=[ f_one_band_energy( array,( B2 [j][0], B2 [j][1]) ) for j in range(len(B2))]# energy bands2 values of f_signal
        E3=[ f_one_band_energy( array,( B3 [j][0], B3 [j][1]) ) for j in range(len(B3))]# energy bands3 values of f_signal
    
        E_one_axis = E1+E2+E3 # list of energy bands values of one f_signal
        
        E_3_axis= E_3_axis + E_one_axis # add values to the global list
    
    return E_3_axis

def f_axial_features_generation(f_window):
    
    
    axial_columns=f_window.columns[0:12] # select frequency axial column names
    axial_df=f_window[axial_columns] # select frequency axial signals in one dataframe
    f_all_axial_features=[] # a global list will contain all frequency axial features values
    
    
    
    for col in range(0,12,3):# iterate throw each group of frequency axial signals in a window
        
        df=axial_df[axial_columns[col:col+3]]  # select each group of 3-axial signals
      
        # mean
        mean_vector                  = mean_axial(df) # 3 values
        # std
        std_vector                   = std_axial(df) # 3 values
        # mad
        mad_vector                   = mad_axial(df) # 3 values
        # max
        max_vector                   = max_axial(df) # 3 values
        # min
        min_vector                   = min_axial(df) # 3 values
        # sma
        sma_value                    = f_sma_axial(df)
        # energy
        energy_vector                = f_energy_axial(df)# 3 values
        # IQR
        IQR_vector                   = IQR_axial(df) # 3 values
        # entropy
        entropy_vector               = entropy_axial(df) # 3 values
        # max_inds
        max_inds_vector              = f_max_Inds_axial(df)# 3 values
        # mean_Freq
        mean_Freq_vector             = f_mean_Freq_axial(df)# 3 values
        # skewness and kurtosis
        skewness_and_kurtosis_vector = f_skewness_and_kurtosis_axial(df)# 6 values
        # bands energy
        bands_energy_vector          = f_all_bands_energy_axial(df) # 42 values

        # append all values of each 3-axial signals in a list
        f_3axial_features = mean_vector +std_vector + mad_vector + max_vector + min_vector + [sma_value] + energy_vector + IQR_vector + entropy_vector + max_inds_vector + mean_Freq_vector + skewness_and_kurtosis_vector + bands_energy_vector

        f_all_axial_features = f_all_axial_features+ f_3axial_features # add features to the global list
        
    return f_all_axial_features

# Functions used to generate frequency magnitude features

# sma
def f_sma_mag(mag_column):
    
    array=np.array(mag_column)
    sma_value=float((abs(array)/math.sqrt(len(mag_column))).sum()) # sma of one mag f_signals
    
    return sma_value

# energy
def f_energy_mag(mag_column):
    
    array=np.array(mag_column)
    # spectral energy value
    energy_value=float((array**2).sum()/float(len(array))) # energy value of one mag f_signals
    return energy_value


####### Max Inds and Mean_Freq Functions#######################################


# max_Inds
def f_max_Inds_mag(mag_column):
    
    array=np.array(mag_column)
    
    max_Inds_value =float(freqs[array[1:65].argmax()+1]) # freq value related with max component
    
    return max_Inds_value

# mean freq()
def f_mean_Freq_mag(mag_column):
    
    array=np.array(mag_column)
    
    mean_freq_value = float(np.dot(freqs,array).sum() / float(array.sum())) # weighted sum of one mag f_signal
    
    return  mean_freq_value

###################################################################################

########## Skewness & Kurtosis Functions #######################################

from scipy.stats import skew           # skewness
def f_skewness_mag(mag_column):
    
    array=np.array(mag_column)
    skew_value     = float(skew(array)) # skewness value of one mag f_signal
    return skew_value



from scipy.stats import kurtosis       # kurtosis
def f_kurtosis_mag(mag_column):
    array=np.array(mag_column)
    kurtosis_value = float(kurtosis(array)) # kurotosis value of on mag f_signal

    return kurtosis_value
##################################################################################

def f_mag_features_generation(f_window):
    
    # select frequnecy mag columns : the last 4 columns in f_window
    mag_columns=f_window.columns[-4:]
    mag_columns=f_window[mag_columns]
    
    f_mag_features=[]
    for col in mag_columns: # iterate throw each mag column in f_window
        
        # calculate common mag features and frequency mag features for each column
        mean_value   = mean_mag(mag_columns[col])
        std_value    = std_mag(mag_columns[col])
        mad_value    = mad_mag(mag_columns[col])
        max_value    = max_mag(mag_columns[col])
        min_value    = min_mag(mag_columns[col])
        sma_value    = f_sma_mag(mag_columns[col])
        energy_value = f_energy_mag(mag_columns[col])
        IQR_value    = IQR_mag(mag_columns[col])
        entropy_value= entropy_mag(mag_columns[col])
        max_Inds_value=f_max_Inds_mag(mag_columns[col])
        mean_Freq_value= f_mean_Freq_mag (mag_columns[col])
        skewness_value=  f_skewness_mag(mag_columns[col])
        kurtosis_value = f_kurtosis_mag(mag_columns[col])
        # 13 value per each t_mag_column
        col_mag_values = [mean_value, std_value, mad_value, max_value, 
                          min_value, sma_value, energy_value,IQR_value, 
                          entropy_value, max_Inds_value, mean_Freq_value,
                          skewness_value, kurtosis_value ]
        
        
        f_mag_features= f_mag_features+ col_mag_values # append feature values of one mag column to the global list
    
    # f_mag_features contains 65 values = 13 value (per each t_mag_column) x 4 (f_mag_columns)
    return f_mag_features

def frequency_features_names():
    #Generating Frequency feature names
    
    # frequency axial signal names 
    axial_signals=[
                    ['f_body_acc_X','f_body_acc_Y','f_body_acc_Z'],
                    ['f_body_acc_Jerk_X','f_body_acc_Jerk_Y','f_body_acc_Jerk_Z'],
                    ['f_body_gyro_X','f_body_gyro_Y','f_body_gyro_Z'],
                    ['f_body_gyro_Jerk_X','f_body_gyro_Jerk_Y','f_body_gyro_Jerk_Z'],
                  ]

    # frequency magnitude signals
    mag_signals=['f_body_acc_Mag','f_body_acc_Jerk_Mag','f_body_gyro_Mag','f_body_gyro_Jerk_Mag']


    # features functions names will be applied to f_signals
    f_one_input_features_name1=['_mean()','_std()','_mad()','_max()','_min()']

    f_one_input_features_name2=['_energy()','_iqr()','_entropy()','_maxInd()','_meanFreq()']

    f_one_input_features_name3= ['_skewness()','_kurtosis()']

    f_one_input_features_name4=[
                                '_BE[1-8]','_BE[9-16]','_BE[17-24]','_BE[25-32]',
                                '_BE[33-40]','_BE[41-48]','_BE[49-56]','_BE[57-64]',
                                '_BE[1-16]','_BE[17-32]','_BE[33-48]','_BE[49-64]',
                                '_BE[1-24]','_BE[25-48]'
                               ]
    
    frequency_features_names=[] # global list of frequency features
    
    for columns in axial_signals: # iterate throw each group of 3-axial signals
        
        # iterate throw the first list of features
        for feature in f_one_input_features_name1: 
            for column in columns:# iterate throw each signal name of that group
                newcolumn=column[:-2]+feature+column[-2:] # build the full feature name
                frequency_features_names.append(newcolumn) # add the feature name to the global list
        
        # sma feature name
        sma_column=column[:-2]+'_sma()'
        frequency_features_names.append(sma_column)

        # iterate throw the first list of features
        for feature in f_one_input_features_name2:
            for column in columns:
                newcolumn=column[:-2]+feature+column[-2:]
                frequency_features_names.append(newcolumn)
        
        # iterate throw each signal name of that group
        for column in columns:
            for feature in f_one_input_features_name3: # iterate throw [skewness ,kurtosis]
                newcolumn=column[:-2]+feature+column[-2:] # build full feature name
                frequency_features_names.append(newcolumn) # append full feature names
        
        # same process above will be applied to list number 4
        for column in columns:
            for feature in f_one_input_features_name4:
                newcolumn=column[:-2]+feature+column[-2:]
                frequency_features_names.append(newcolumn)
   
    #################################################################################################################
    # generate frequency mag features names
    for column in mag_signals:# iterate throw each frequency mag signal name
        for feature in f_one_input_features_name1:# iterate throw the first list of features functions names
            frequency_features_names.append(column+feature) # build the full feature name and add it to the global list

        sma_column=column+'_sma()' # build the sma full feature name
        frequency_features_names.append(sma_column) # add it to the global list

        for feature in f_one_input_features_name2:# iterate throw the second list of features functions names
            frequency_features_names.append(column+feature)# build the full feature name and add it to the global list
        
        for feature in f_one_input_features_name3:# iterate throw the third list of features functions names
            frequency_features_names.append(column+feature)# build the full feature name and add it to the global list
    ####################################################################################################################
    
    return frequency_features_names

############### Angles Functions ####################################
from math import acos # inverse of cosinus function
from math import sqrt # square root function

########Euclidian magnitude 3D############
def magnitude_vector(vector3D): # vector[X,Y,Z]
    return sqrt((vector3D**2).sum()) # eulidian norm of that vector

###########angle between two vectors in radian ###############
def angle(vector1, vector2):
    vector1_mag=magnitude_vector(vector1) # euclidian norm of V1
    vector2_mag=magnitude_vector(vector2) # euclidian norm of V2
   
    scalar_product=np.dot(vector1,vector2) # scalar product of vector 1 and Vector 2
    cos_angle=scalar_product/float(vector1_mag*vector2_mag) # the cosinus value of the angle between V1 and V2
    
    # just in case some values were added automatically
    if cos_angle>1:
        cos_angle=1
    elif cos_angle<-1:
        cos_angle=-1
    
    angle_value=float(acos(cos_angle)) # the angle value in radian
    return angle_value # in radian.

################## angle_features ############################
def angle_features(t_window): # it returns 7 angles per window
    angles_list=[]# global list of angles values
    
    # mean value of each column t_body_acc[X,Y,Z]
    V2_columns=['t_grav_acc_X','t_grav_acc_Y','t_grav_acc_Z']
    V2_Vector=np.array(t_window[V2_columns].mean()) # mean values
    
    # angle 0: angle between (t_body_acc[X.mean,Y.mean,Z.mean], t_gravity[X.mean,Y.mean,Z.mean])
    V1_columns=['t_body_acc_X','t_body_acc_Y','t_body_acc_Z']
    V1_Vector=np.array(t_window[V1_columns].mean()) # mean values of t_body_acc[X,Y,Z]
    angles_list.append(angle(V1_Vector, V2_Vector)) # angle between the vectors added to the global list
    
    # same process is applied to ither signals
    # angle 1: (t_body_acc_jerk[X.mean,Y.mean,Z.mean],t_gravity[X.mean,Y.mean,Z.mean]
    V1_columns=['t_body_acc_jerk_X','t_body_acc_jerk_Y','t_body_acc_jerk_Z']
    V1_Vector=np.array(t_window[V1_columns].mean())
    angles_list.append(angle(V1_Vector, V2_Vector))
    
    # angle 2: (t_body_gyro[X.mean,Y.mean,Z.mean],t_gravity[X.mean,Y.mean,Z.mean]
    V1_columns=['t_body_gyro_X','t_body_gyro_Y','t_body_gyro_Z']
    V1_Vector=np.array(t_window[V1_columns].mean())
    angles_list.append(angle(V1_Vector, V2_Vector))
    
    # angle 3: (t_body_gyro_jerk[X.mean,Y.mean,Z.mean],t_gravity[X.mean,Y.mean,Z.mean]
    V1_columns=['t_body_gyro_jerk_X','t_body_gyro_jerk_Y','t_body_gyro_jerk_Z']
    V1_Vector=np.array(t_window[V1_columns].mean())
    angles_list.append(angle(V1_Vector, V2_Vector))
    #################################################################################
    
    # V1 vector in this case is the X axis itself [1,0,0]
    # angle 4: ([X_axis],t_gravity[X.mean,Y.mean,Z.mean])   
    V1_Vector=np.array([1,0,0])
    angles_list.append(angle(V1_Vector, V2_Vector))
    
    # V1 vector in this case is the Y axis itself [0,1,0]
    # angle 5: ([Y_acc_axis],t_gravity[X.mean,Y.mean,Z.mean]) 
    V1_Vector=np.array([0,1,0])
    angles_list.append(angle(V1_Vector, V2_Vector))
    
    # V1 vector in this case is the Z axis itself [0,0,1]
    # angle 6: ([Z_acc_axis],t_gravity[X.mean,Y.mean,Z.mean])
    V1_Vector=np.array([0,0,1])
    angles_list.append(angle(V1_Vector, V2_Vector))
    
    return angles_list

angle_columns=['angle0()','angle1()','angle2()','angle3()','angle4()','angle5()','angle6()']

all_columns=time_features_names()+frequency_features_names()+angle_columns+['activity_Id','user_Id']

def Dataset_Generation_PipeLine(t_dic,f_dic):
    # t_dic is a dic contains time domain windows
    # f_dic is a dic contains frequency domain windows
    # f_dic should be the result of applying fft to t_dic
    
    final_Dataset=pd.DataFrame(data=[],columns= all_columns) # build an empty dataframe to append rows
    
    for i in range(len(t_dic)): # iterate throw each window

        # t_window and f_window should have the same window id included in their keys
        t_key=sorted(t_dic.keys() )[i] # extract the key of t_window
        f_key=sorted(f_dic.keys() )[i] # extract the key of f_window 
        
        t_window=t_dic[t_key] # extract the t_window
        f_window=f_dic[f_key] # extract the f_window

        window_user_id= int(t_key[-8:-6]) # extract the user id from window's key
        window_activity_id=int(t_key[-2:]) # extract the activity id from the windows key

        # generate all time features from t_window 
        time_features = t_axial_features_generation(t_window) + t_mag_features_generation(t_window)
        
        # generate all frequency features from f_window
        frequency_features = f_axial_features_generation(f_window) + f_mag_features_generation(f_window)
        
        # Generate addtional features from t_window
        additional_features= angle_features(t_window)
        # concatenate all features and append the activity id and the user id
        row= time_features + frequency_features + additional_features + [int(window_activity_id),int(window_user_id)]
        
        # go to the first free index in the dataframe
        free_index=len(final_Dataset)
        
        # append the row
        final_Dataset.loc[free_index]= row
        
    return final_Dataset # return the final dataset