#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 13:19:08 2020

Homework answers for lesson 4 of "Friends of Tracking" #FoT

Data can be found at: https://github.com/metrica-sports/sample-data

@author: Laurie Shaw (@EightyFivePoint)
"""

import Metrica_IO as mio
import Metrica_Viz as mviz
import pandas as pd
import numpy as np

# set up initial path to data
DATADIR = 'C:\github\sample-data\data'
game_id = 2 # let's look at sample match 2

# read in the event data
events = mio.read_event_data(DATADIR,game_id)

# count the number of each event type in the data
print( events['Type'].value_counts() )

# Bit of housekeeping: unit conversion from metric data units to meters
events = mio.to_metric_coordinates(events)

# Get events by team
home_events = events[events['Team']=='Home']
away_events = events[events['Team']=='Away']

# Frequency of each event type by team
home_events['Type'].value_counts()
away_events['Type'].value_counts()

# Get all shots
shots = events[events['Type']=='SHOT']
home_shots = home_events[home_events.Type=='SHOT']
away_shots = away_events[away_events.Type=='SHOT']

# Look at frequency of each shot Subtype
home_shots['Subtype'].value_counts()
away_shots['Subtype'].value_counts()

# Look at the number of shots taken by each home player
print( home_shots['From'].value_counts() )

# Get the shots that led to a goal
home_goals = home_shots[home_shots['Subtype'].str.contains('-GOAL')].copy()
away_goals = away_shots[away_shots['Subtype'].str.contains('-GOAL')].copy()

# Add a column event 'Minute' to the data frame
home_goals['Minute'] = home_goals['Start Time [s]']/60.
away_goals['Minute'] = away_goals['Start Time [s]']/60.

# Plot the first goal
fig,ax = mviz.plot_pitch()
ax.plot( events.loc[198]['Start X'], events.loc[198]['Start Y'], 'ro' )
ax.annotate("", xy=events.loc[198][['End X','End Y']], xytext=events.loc[198][['Start X','Start Y']], alpha=0.6, arrowprops=dict(arrowstyle="->",color='r'))

# plot passing move in run up to goal
mviz.plot_events( events.loc[190:198], indicators = ['Marker','Arrow'], annotate=True )

#### TRACKING DATA ####

# READING IN TRACKING DATA
tracking_home = mio.tracking_data(DATADIR,game_id,'Home')
tracking_away = mio.tracking_data(DATADIR,game_id,'Away')

# Look at the column namems
print( tracking_home.columns )

# Convert positions from metrica units to meters 
tracking_home = mio.to_metric_coordinates(tracking_home)
tracking_away = mio.to_metric_coordinates(tracking_away)

# Plot some player trajectories (players 11,1,2,3,4)
fig,ax = mviz.plot_pitch()
ax.plot( tracking_home['Home_11_x'].iloc[:1500], tracking_home['Home_11_y'].iloc[:1500], 'r.', MarkerSize=1)
ax.plot( tracking_home['Home_1_x'].iloc[:1500], tracking_home['Home_1_y'].iloc[:1500], 'b.', MarkerSize=1)
ax.plot( tracking_home['Home_2_x'].iloc[:1500], tracking_home['Home_2_y'].iloc[:1500], 'g.', MarkerSize=1)
ax.plot( tracking_home['Home_3_x'].iloc[:1500], tracking_home['Home_3_y'].iloc[:1500], 'k.', MarkerSize=1)
ax.plot( tracking_home['Home_4_x'].iloc[:1500], tracking_home['Home_4_y'].iloc[:1500], 'c.', MarkerSize=1)

# plot player positions at ,atckick-off
KO_Frame = events.loc[0]['Start Frame']
fig,ax = mviz.plot_frame( tracking_home.loc[KO_Frame], tracking_away.loc[KO_Frame] )

# PLOT POISTIONS AT GOAL
fig,ax = mviz.plot_events( events.loc[198:198], indicators = ['Marker','Arrow'], annotate=True )
goal_frame = events.loc[198]['Start Frame']
fig,ax = mviz.plot_frame( tracking_home.loc[goal_frame], tracking_away.loc[goal_frame], figax = (fig,ax) )

# Plot the passes and shot leading up to the second goal of the home-team
goal_index = home_goals.index[1]
fig, ax2 = mviz.plot_events(events.loc[goal_index:goal_index], indicators = ['Marker','Arrow'], annotate=True)
for j in range(home_goals.index[1]-3,home_goals.index[1]):
    if events.loc[j][['End X']].isnull().values[0]:
        fig, ax2 = mviz.plot_events(events.loc[j:j], figax=(fig,ax2), field_dimen = (106.0,68), indicators = ['Marker'], color='r', marker_style = 'o', alpha = 0.5, annotate=True)
    else:
        fig, ax2 = mviz.plot_events(events.loc[j:j], figax=(fig,ax2), field_dimen=(106.0, 68), indicators=['Marker','Arrow'], color='r',
                         marker_style='o', alpha=0.5, annotate=True)

# Plot the passes and shot leading up to the third goal of the home-team
goal_index = home_goals.index[2]
fig, ax3 = mviz.plot_events(events.loc[goal_index:goal_index], indicators = ['Marker','Arrow'], annotate=True)
for j in range(goal_index-3,goal_index):
    if events.loc[j][['End X']].isnull().values[0]:
        fig, ax3 = mviz.plot_events(events.loc[j:j], figax=(fig,ax3), field_dimen=(106.0,68), indicators = ['Marker'], color='r', marker_style = 'o', alpha = 0.5, annotate=True)
    else:
        fig, ax3 = mviz.plot_events(events.loc[j:j], figax=(fig,ax3), field_dimen=(106.0, 68), indicators=['Marker','Arrow'], color='r',
                         marker_style='o', alpha=0.5, annotate=True)

# Plot all the shots by Player 9 of the home team. Use a different symbol and transparency (alpha) for shots that resulted in goals
home_shots_Player9 = home_shots[home_shots['From'] == 'Player9'].copy() # create a new data frame identical to home_shots but only with the shots from Player9
fig,ax4 = mviz.plot_pitch()
for jj, rows in home_shots_Player9.iterrows():
    if home_shots_Player9.loc[jj][['Subtype']].str.contains('-GOAL').values[0]:
        fig, ax4 = mviz.plot_events(home_shots_Player9.loc[jj:jj], figax=(fig, ax4), field_dimen=(106.0, 68), indicators=['Marker', 'Arrow'], color='r', marker_style='o', alpha=1, annotate=True)
    else:
        fig, ax4 = mviz.plot_events(home_shots_Player9.loc[jj:jj], figax=(fig, ax4), field_dimen=(106.0, 68), indicators=['Marker', 'Arrow'], color='b', marker_style='*', alpha=0.2, annotate=True)

# Plot the position of all players at Player 9's goal
home_goals_Player9 = home_shots_Player9[home_shots_Player9['Subtype'].str.contains('-GOAL')].copy()
goal_index = home_goals_Player9.index[0]
fig,ax5 = mviz.plot_events(home_goals_Player9.loc[goal_index:goal_index], indicators = ['Marker','Arrow'], annotate=True)
goal_frame = events.loc[goal_index]['Start Frame']
fig,ax5 = mviz.plot_frame(tracking_home.loc[goal_frame], tracking_away.loc[goal_frame], figax = (fig,ax5))

# Calculate how far each player ran
tracking_home_nan20 = (tracking_home.fillna(0)).drop(columns=['Period', 'Time [s]', 'ball_x', 'ball_y']) # replacing all the NaN by 0 in this data frame and remove the columns with the specified names
tracking_away_nan20 = (tracking_away.fillna(0)).drop(columns=['Period', 'Time [s]', 'ball_x', 'ball_y']) # replacing all the NaN by 0 in this data frame and remove the columns with the specified names
home_dist_temp = [0]*len(tracking_home_nan20.columns)
away_dist_temp = [0]*len(tracking_away_nan20.columns)
k = 0
for columnName, columnData in tracking_home_nan20.iteritems(): # for cycle to sum the distance in each axis (x and y) for each player in tracking_home_nan20
    home_dist_temp[k] = abs(tracking_home_nan20.loc[:,columnName].diff()).fillna(0)#.sum()
    #for rowName in range(2, len(tracking_home_nan20)):
     #   home_dist_temp[k] = home_dist_temp[k] + abs(tracking_home_nan20.loc[rowName,columnName] - tracking_home_nan20.loc[rowName-1,columnName])
    k = k + 1
kk = 0
for columnName, columnData in tracking_away_nan20.iteritems(): # for cycle to sum the distance in each axis (x and y) for each player in tracking_away_nan20
    away_dist_temp[kk] = abs(tracking_away_nan20.loc[:, columnName].diff()).fillna(0)#.sum()
    #for rowName in range(2, len(tracking_away_nan20)):
     #   away_dist_temp[kk] = away_dist_temp[kk] + abs(tracking_away_nan20.loc[rowName,columnName] - tracking_away_nan20.loc[rowName-1,columnName])
    kk = kk + 1

home_dist = [0]*int(len(tracking_home_nan20.columns)/2) # initialize a list row with half the number of columns in the data frame tracking_home_nan20
away_dist = [0]*int(len(tracking_away_nan20.columns)/2) # initialize a list row with half the number of columns in the data frame tracking_away_nan20

j = 0
for jj in range(0,len(home_dist_temp),2): # for cycle to calculate: distance = sqrt((|x_i+1 - x_i|)^2 + (|y_i+1 - y_i|)^2))
    home_dist[j] = np.sqrt(home_dist_temp[jj]**2 + home_dist_temp[jj+1]**2).sum()
    j = j + 1
#home_dist = pd.DataFrame.transpose(pd.DataFrame(data = home_dist, index = tracking_home_nan20.columns[0::2])) # transform the previously initialized list in a data frame and transpose it, including the names of the even columns in tracking_home_nan20

j = 0
for jj in range(0,len(away_dist_temp),2): # for cycle to calculate: distance = sqrt((|x_i+1 - x_i|)^2 + (|y_i+1 - y_i|)^2))
    away_dist[j] = np.sqrt(away_dist_temp[jj]**2 + away_dist_temp[jj+1]**2).sum()
    j = j + 1
#away_dist = pd.DataFrame.transpose(pd.DataFrame(data = away_dist, index = tracking_away_nan20.columns[0::2])) # transform the previously initialized list in a data frame and transpose it, including the names of the even columns in tracking_away_nan20
# END
