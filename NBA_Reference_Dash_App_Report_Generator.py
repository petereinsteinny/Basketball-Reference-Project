#!/usr/bin/env python
# coding: utf-8

# In[19]:


#importing packages
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


#Get list of active players
def refresh_active_players():
    global players_df
    players_df=pd.read_html('https://www.basketball-reference.com/leagues/NBA_2023_per_game.html')[0]
    players_df=players_df[players_df['Player']!='Player']
    players_df=players_df.iloc[:,1:5]
    players_df.set_index('Player',inplace=True)
    #Get links for all active players to be used in querying
    urls = 'https://www.basketball-reference.com/leagues/NBA_2023_per_game.html'
    grab = requests.get(urls)
    soup = BeautifulSoup(grab.text, 'html.parser')
    soup=soup.find('table')

    links=[]

    for link in soup.find_all("a"):
        data = link.get('href')
        if data[:9] =='/players/'and data[-5:]=='.html':
            url='https://www.basketball-reference.com'+data
            links.append(url)

    players_df['Link']=links

    players_df.to_csv('Active Players.csv')


# In[21]:


#Refresh player stats
def refresh_player_stats():
    global unique_players,player_queries_standard,formatted_player_queries_advanced
    #Looping through the unique players list to populate data for each player.
    for player in unique_players:
        time.sleep(1.75)
        #Instantiating each Player so that their data can be generated and appended
        this_player=Player(player)
           
    player_queries_standard.to_csv('Unformatted_Player_Queries_Standard.csv')
    player_queries_advanced.to_csv('Unformatted_Player_Queries_Advanced.csv')
    formatted_player_queries_standard.to_csv('Formatted_Player_Queries_Standard.csv')
    formatted_player_queries_advanced.to_csv('Formatted_Player_Queries_Advanced.csv')


# In[22]:


#Refresh Rosters
def refresh_rosters():
    global players_df, rosters_df
    
    #Getting a list of unique teams from the players_df. Getting rid of 'TOT' as this indicates 'Total' as opposed to an individual team
    unique_teams=players_df[(players_df['Tm']!='TOT') & (players_df['Tm']!='Tm')]['Tm'].sort_values().unique().tolist()

    for team in unique_teams:
        #Instantiating each Player so that their data can be generated and appended
        time.sleep(1.75)
        this_team=Team(team)
        
    rosters_df.to_csv('Active Rosters.csv')


# In[23]:


#Function to perform all refreshing tasks
def refresh_all():
    global player_queries_standard, player_queries_advanced,formatted_player_queries_standard,formatted_player_queries_advanced, rosters_df
    print("I'm working...")
    
    player_queries_standard=[]
    player_queries_advanced=[]
    formatted_player_queries_standard=[]
    formatted_player_queries_advanced=[]
    rosters_df=[]
    
    refresh_active_players()
    print("I'm done refreshing active players")
    refresh_player_stats()
    print("I'm done refreshing player stats")
    time.sleep(1.75)
    refresh_rosters()
    print("I'm done refreshing rosters")
    print("I'm done with all tasks")
    

#schedule.every().day.at("15:55").do(refresh_all)

#while True:
#    schedule.run_pending()
#    time.sleep(1) # wait one minute


# In[24]:


#Creating player class to inherit attributes and functions


positions=['PG','SG','SF','PF','C']

class Player:
    
    def __init__(self,name):
        #Populating tables with player data at instantiation
        global player_queries_standard, player_queries_advanced, formatted_player_queries_standard, formatted_player_queries_advanced
        self.name=name
        if len(player_queries_standard)==0:
            queries=self.get_reports()
            player_queries_standard=queries[0]
            player_queries_advanced=queries[1]
            formatted_player_queries_standard=Player(self.name).format_per_game('Standard')
            formatted_player_queries_advanced=Player(self.name).format_per_game('Advanced')
        elif name not in player_queries_standard['Player Name'].to_list():
            queries=self.get_reports()
            player_queries_standard=pd.concat([player_queries_standard,queries[0]],axis=0)
            player_queries_advanced=pd.concat([player_queries_advanced,queries[1]],axis=0)
            formatted_player_queries_standard=pd.concat([formatted_player_queries_standard,Player(self.name).format_per_game('Standard')],axis=0)
            formatted_player_queries_advanced=pd.concat([formatted_player_queries_advanced,Player(self.name).format_per_game('Advanced')],axis=0)
            
    def __str__(self):
        #Function to print formatted description when printing Player
        name=self.name
        pos=self.get_position()
        age=self.get_age()
        tm=self.get_team()
        
        return "Player: {}, Position: {}, Age: {}, Most Recent Team: {}".format(name,pos,age,tm)
    
    def get_info(self,field):
        global player_queries_standard
        #Querying the formatted player_queries_standard table to get info from columns
        df=player_queries_standard[player_queries_standard['Player Name']==self.name]
        #The most recent season will be the one beneath the Career row
        most_recent=df[df['Season']=='Career'].index.to_numpy().sum()-1
        
        return df.loc[most_recent][field]
    
    def get_player_url(self):
        #Getting player link from the players_df to use in information retrieval
        val=players_df.loc[self.name]['Link']
        
        if type(val) is str:
            return val
        else:
            return val[0]
    
    def get_age(self):
        #Function to get player age
        return self.get_info('Age')
    
    def get_position(self):
        #Function to get most recent player position
        return self.get_info('Pos')
    
    def get_positions(self,df):
        global positions
        #function to get a unique list of positions played by a player. This output is used in peer-group filtering
        #Replacing comma-separated positions lists (e.g. PF,C) with | for splitting
        player_positions=df['Pos'].dropna().apply(lambda x: x.replace(",","|")).unique()
        all_positions=[]
        
        for i in player_positions:
            if i not in all_positions:
                if '|' not in i:
                    all_positions.append(i)
                else:
                    vals=i.split('|')
                    for sub_val in vals:
                        if sub_val not in all_positions:
                            all_positions.append(i)
        return all_positions
    
    
    def get_team(self):
        #Function to get player team
        return self.get_info('Tm')
    
    def get_reports(self):
        global positions
        #Querying Basketball Reference to get the Standard (Per Game) and Advanced tables
        req=requests.get(self.get_player_url()).text
        #Creating one big soup to house all tables
        orig_soup=BeautifulSoup(req,'html.parser')
        #Filtering the original soup for the Standard data table
        standard_soup=orig_soup.find(id='div_per_game')
        standard_table=standard_soup.find_all('table')
        standard_df = pd.read_html(str(standard_table))[0]
        
        #Creating the player's unique list of positions using the get_positions function
        unique_positions=self.get_positions(standard_df)
        
        #Setting position columns to be true if the player plays that position and false if they don't
        for pos in positions:
            if pos in unique_positions:
                standard_df[pos]=True
            else:
                standard_df[pos]=False
        
        #Adding a Player Name column so the Standard and Advanced tables can be queried (thus using caching and eliminating need to query website again)
        standard_df['Player Name']=self.name
        
        #Filtering the original soup for the Advanced data table  
        advanced_soup=orig_soup.find(id='div_advanced')
        advanced_table=advanced_soup.find_all('table')
        advanced_df = pd.read_html(str(advanced_table))[0]
        advanced_df[positions]=standard_df[positions]
        advanced_df['Player Name']=self.name
        
        #Ensuring that all columns converted to numeric where possible in both Standard and Advanced tables
        standard_df=standard_df.apply(pd.to_numeric,errors='ignore')
        advanced_df=advanced_df.apply(pd.to_numeric,errors='ignore')
        return standard_df, advanced_df
    
    def format_per_game(self,report_type):
        global player_queries_standard, player_queries_advanced
        
        if report_type=='Standard':
            df=player_queries_standard
        elif report_type=='Advanced':
            df=player_queries_advanced
            
        df=df[df['Player Name']==self.name]
        
        df=df.dropna(axis=1, how='all')
        
        #Getting rid of overseas data
        df=df[df['Lg']=='NBA']
        
        #Ensuring only seasons being included in dataframe
        
        df=df.iloc[0:df.query('Season == "Career"').index[0]]
        
        #Ensuring that all player data per season is consolidated into one row (no splits across teams)
        multi_team_df=df.copy().groupby('Season').count()['Tm']
        df=df[df.apply(lambda x: True if int(multi_team_df.loc[x['Season']])==1 or x['Tm']=='TOT'
              else False
            ,axis=1)]
        return df.apply(pd.to_numeric,errors='ignore')
    
    def get_formatted_per_game(self,report_type):
        #Creating function to query formatted data tables for player stats w/ caching
        global positions, formatted_player_queries_standard, formatted_player_queries_advanced
        
        if report_type=='Standard':
            df=formatted_player_queries_standard
        elif report_type=='Advanced':
            df=formatted_player_queries_advanced
            
        return df[df['Player Name']==self.name]
        
    
    def get_full_peer_group(self):
        #Creating function to retrieve a base player's peer group to be used in the clustering analysis
        global formatted_player_queries_advances, positions
        player_age=self.get_age()
        #Setting split model where if a player is younger than 32, return a peer group of player's older than them. Else, include all players since their career arc is near complete (e.g. Udonis Haslem)
        if player_age<32:
            comparables_df=formatted_player_queries_advanced[formatted_player_queries_advanced['Age']>player_age]
        else:
            comparables_df=formatted_player_queries_advanced.copy()
        
        #Setting up a positions_array so that the comparables_df can be filtered for peers that shared any of the same positions as the base player (sum positions_array>0)
        positions_array=self.get_formatted_per_game('Advanced')[positions].iloc[0]
        positions_array=positions_array[positions_array== True].index

        players_array=comparables_df[comparables_df[positions_array].sum(axis=1)>0]['Player Name'].unique()

        return players_array
    
    def get_top_3_cluster(self):
        global ws_df
        #Creating a function to filter the full peer group to return up to 3 most similar players
        #Getting peer group using get_full_peer_group function
        comparable_list=self.get_full_peer_group()
        #Copying the ws_df (Index: Player_name, Columns: Age, Rows: Win Shares)
        comparable_df=ws_df.copy()
        player_name=self.name
        #Dropping out the base player's name so that it isn't returned as one of the top 3 most similar players
        comparable_df=comparable_df.drop(player_name)
        #Filtering the comparable_df to only include player values from the peer_list
        comparable_df=comparable_df[comparable_df.index.isin(comparable_list)]
        
        #Setting up an array of the player's W/S through the present to be used in predicting their peer group
        player_age=int(self.get_age())
        prediction_list=ws_df.loc[player_name,:player_age]
        prediction_list=np.array(prediction_list).reshape(1,-1)
        #Filtering the ws_df to include comparables player data through the base_player's age to be used as the X variable
        min_age=int(ws_df.columns[0])
        x_range=list(range(min_age,player_age+1))
        X=comparable_df[x_range]
        #Setting the Y variable as the career-to-date WS total
        Y=comparable_df['Career']
        
        #Fitting asnd transforming X data
        Clus_dataSet = StandardScaler().fit_transform(X)
        
        #Instantiating the k_means model. 10 clusters was determined after using the elbow-method to find the point of declining marginal inertia
        k_means = KMeans(init = "k-means++", n_clusters = 10, n_init = 20)
        #Fitting the k_means model
        k_means.fit(X)
        #Retrieving the labels from the k_means model and populating them into the comparable_df
        labels=k_means.labels_
        comparable_df['label']=labels
        
        #Using the k_means model to predict which class the base player should belong to
        cluster_num=k_means.predict(prediction_list).sum()
        
        #Getting the list of all players who share the same cluster number as predicted for the base player
        player_cluster=comparable_df[comparable_df['label']==cluster_num].index.to_list()
        #Getting a filtered comparble_df containing only the player_cluster
        player_cluster_df=comparable_df.loc[player_cluster,:player_age]

        #Converting the player_cluster_df into an array
        player_cluster_arr=np.array(player_cluster_df)
        #Creating the base player array to compare against the player cluster array
        base_player_arr=np.array(ws_df.loc[player_name,:player_age])

        #Checking the Euclidian distance between the base_player_arr and the player_cluster_arr
        dists=[]

        for comp_player in player_cluster_arr:
            sum_sq = np.sum(np.square(base_player_arr - comp_player))
            if len(dists)==0:
                dists=[sum_sq]
            else:
                dists.append(sum_sq)
        #Populating the player_cluster_df with their Euclidian distances
        player_cluster_df['Distance']=dists
        #Getting the top_5 lowest distances from the base player and converting to a list
        top_3_neigh=player_cluster_df.sort_values(by='Distance',axis=0,ascending=True).head(3).index.to_list()

        return top_3_neigh


# In[25]:


#Creating a team class to be used in returning active rosters in the dash player list dropdown
class Team():
    def __init__(self,name):
        #Instantiating the Team class through retrieving their roster using the get_rosters function
        global rosters_df
        self.name=name
        self.url='https://www.basketball-reference.com/teams/'+name+'/2023.html'
        if len(rosters_df)==0:
            rosters_df=self.get_rosters()
        elif name not in rosters_df['Tm'].tolist():
            this_roster=self.get_rosters()
            rosters_df=pd.concat([rosters_df,this_roster],axis=0)
            
    def get_rosters(self):
        #Creating a function to parse the team roster from their html page
        req=requests.get(self.url).text
        soup=BeautifulSoup(req,'html.parser')
        soup=soup.find(id='div_roster')
        table=soup.find_all('table')
        df = pd.read_html(str(table))[0]
        #Basketball reference appends (TW) to a player's name to indicate they are a two-way player. The below removes this convention.
        df['Player']=df['Player'].apply(lambda x: x.replace(' (TW)',''))
        #Adding Team name into rosters_df for future filtering
        df['Tm']=self.name
        
        #Getting all player links from the roster's page
        links=[]

        for link in soup.find_all("a"):
            data = link.get('href')
            if data[:9] =='/players/'and data[-5:]=='.html':
                url='https://www.basketball-reference.com'+data
                links.append(url)
        #Populating the df with player links
        df['Link']=links
        #Convering all values in the df with numbers where possible
        df=df.apply(pd.to_numeric,errors='ignore')
        return df
    
    def get_roster(self):
        #Creating a function to dynamically get a team's roster from the rosters_df
        return rosters_df[rosters_df['Tm']==self.name]


# In[1]:


refresh_all()

