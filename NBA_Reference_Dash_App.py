#!/usr/bin/env python
# coding: utf-8

# In[110]:


#importing packages
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels
import matplotlib.pyplot as plt
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input,Output, State
import plotly.graph_objects as go
import plotly.express as px
from dash import no_update
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[111]:


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


# In[112]:


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


# In[113]:


players_df=pd.read_csv('https://github.com/petereinsteinny/Basketball-Reference-Project/blob/main/Active%20Players.csv?raw=true')
players_df.set_index('Player',inplace=True)
player_queries_standard=pd.read_csv('https://github.com/petereinsteinny/Basketball-Reference-Project/blob/main/Unformatted_Player_Queries_Standard.csv?raw=true')
player_queries_advanced=pd.read_csv('https://github.com/petereinsteinny/Basketball-Reference-Project/blob/main/Unformatted_Player_Queries_Advanced.csv?raw=true')
formatted_player_queries_standard=pd.read_csv('https://github.com/petereinsteinny/Basketball-Reference-Project/blob/main/Formatted_Player_Queries_Standard.csv?raw=true')
formatted_player_queries_standard.drop('Unnamed: 0',axis=1, inplace=True)
formatted_player_queries_advanced=pd.read_csv('https://github.com/petereinsteinny/Basketball-Reference-Project/blob/main/Formatted_Player_Queries_Advanced.csv?raw=true')
formatted_player_queries_advanced.drop('Unnamed: 0',axis=1, inplace=True)
rosters_df=pd.read_csv('https://github.com/petereinsteinny/Basketball-Reference-Project/blob/main/Active%20Rosters.csv?raw=true')


# In[114]:



#Dynamically retrieving the last season so the portion of regular season games played (82/n). This transforms the current season to date stats into a full-season value.
last_season=sorted(formatted_player_queries_advanced['Season'].to_list())[-1]
games_played_ratio=82/(formatted_player_queries_advanced[formatted_player_queries_advanced['Season']==last_season]['G'].max())

#Annualizing current year Performance
formatted_player_queries_advanced['OWS']=formatted_player_queries_advanced.apply(lambda x: x['OWS']*games_played_ratio if x['Season']==last_season
            else x['OWS'],axis=1)
formatted_player_queries_advanced['DWS']=formatted_player_queries_advanced.apply(lambda x: x['DWS']*games_played_ratio if x['Season']==last_season
            else x['DWS'],axis=1)
formatted_player_queries_advanced['WS']=formatted_player_queries_advanced.apply(lambda x: x['WS']*games_played_ratio if x['Season']==last_season
            else x['WS'],axis=1)

#Creating a Win Shares Data Frame with (Index: Player Name, Columns: Player Age, Rows: WS)
ws_df=formatted_player_queries_advanced.copy()

ws_df=ws_df.pivot(index='Player Name',columns='Age',values='WS')
#Overriding NaN values indicating the player didn't log any NBA minutes at X age with 0.
ws_df=ws_df.replace(np.nan,0)
#Creating a career column to house all Career production
ws_df['Career']=ws_df.sum(axis=1)


# In[115]:


unique_players=players_df.index.sort_values().unique().tolist()
#Converting the unique_players list into the dropdown format to be used in Dash
all_players_choices=choices=[{'label':i, 'value':i} for i in unique_players]

#Getting a list of unique teams from the players_df. Getting rid of 'TOT' as this indicates 'Total' as opposed to an individual team
unique_teams=players_df[(players_df['Tm']!='TOT') & (players_df['Tm']!='Tm')]['Tm'].sort_values().unique().tolist()

unique_teams.insert(0,'ALL')
#Creating the Team dropdown list from unique_teams for the Dash
teams_choices=[{'label':i, 'value':i} for i in unique_teams]

test_player=Player(players_df.index[0])

df=test_player.get_formatted_per_game('Standard')
standard_cols=df.select_dtypes(include=np.number).drop('Age',axis=1).columns.to_list()

df=test_player.get_formatted_per_game('Advanced')
advanced_cols=df.select_dtypes(include=np.number).drop('Age',axis=1).columns.to_list()


# In[121]:


#Wrote project description to be used in Dash
project_description=project_description='The below dashboard uses the K-Means Clustering Machine Learning technique to return up to 3 comparable active NBA players on a Win Share (WS) by Age basis. These comparable players are sorted from most-to-least similar in the legend for each chart. Basketball Reference defines this metric as a "player statistic which attempts to divvy up credit for team success to the individuals on the team". For more information on this project, or to reach out to me with any questions, please refer to my:'


# In[1]:

#Creating dictionary to house player cluster outputs
player_clusters={}

#Instantiating Dash app
app=dash.Dash(__name__)
server = app.server

#Checking callbacks to ensure referenced IDs exist and props are valid
app.config.suppress_callback_exceptions = True


#Setting up the app layout
app.layout = html.Div(children=[ 
    html.Div(children=[
        #Creating the header dropdown here
        html.H1('Basketball Career Visualizer', 
            style={'textAlign': 'center',
                   'color': '#503D36',
                   'font-size': 32,
                    'margin':5}),
        #Creating the subheader for my name
        html.H2('By Peter Einstein, CFA', 
            style={'textAlign': 'center',
                   'color': '#503D36',
                   'font-size': 26,
                  'font-weight': 'normal',
                  'margin':15}),
        #Creating the project description subheader here
        html.H3(project_description,
                style={'textAlign': 'left',
                   'color': '#503D36',
                   'font-size': 20,
                  'font-weight': 'normal',
                      'margin':5}),
        #Creating GitHub Link
        html.A("GitHub Project Repository", href='https://github.com/petereinsteinny/Basketball-Reference-Project', target="_blank",
                style={'textAlign': 'left',
                       'color':'blue',
                   'font-size': 20}),
        html.Br(),
        #Creating LinkedIn Link
        html.A("LinkedIn", href='https://www.linkedin.com/in/peter-einstein-cfa/', target="_blank",
                style={'textAlign': 'left', 
                       'color':'blue',
                   'font-size': 20}),
        html.Br(),
        html.Div(children=[
            #Creating the Team dropdown header and Dropdown here. Setting it to 50% and inline-block so it goes side by side with the Player Dropdown
            html.H2('Team Dropdown:', 
                    style={'textAlign': 'left',
                           'color': '#503D36',
                           'font-size': 20}),
            dcc.Dropdown(id='team-dropdown',
                     options=[{'label':i, 'value':i} for i in unique_teams],
                     value='ALL',
                     placeholder="Select a Team Here",
                     searchable=False)
        ],style={'width': '50%', 'display': 'inline-block'}),
        
        #Creating the Player dropdown header and Dropdown here. Setting it to 50% and inline-block so it goes side by side with the Team Dropdown
        html.Div(children=[
            html.H2('Player Dropdown:', 
                    style={'textAlign': 'left',
                           'color': '#503D36',
                           'font-size': 20}),
            dcc.Dropdown(id='player-dropdown',
                         options=[{'label':i, 'value':i} for i in unique_players],
                         value=unique_players[0],
                         placeholder="Select a Player Here",
                         searchable=True),
        ],style={'width': '50%', 'display': 'inline-block'}),
        html.Br()
    ]),
    
    #Creating the Standard Statistics header and dropdown here as well as the chart. Setting it to 50% and inline-block so it goes side by side with the Advanced Dropdown
    html.Div(children=[
        html.H2('Standard Statistics:', style={'margin-right': '2em'}),
        
        dcc.Dropdown(id='standard-dropdown',
                 options=[{'label':i, 'value':i} for i in standard_cols],
                 value='PTS',
                 placeholder="Select a Standard Statistic Here",
                 searchable=False
                ),
        dcc.Graph(id='standard-chart', 
                       style={'display': 'flex','flex-direction':'column'}),

    ],style={'width': '50%', 'display': 'inline-block'}),
    
    #Creating the Advanced Statistics header and dropdown here as well as the chart. Setting it to 50% and inline-block so it goes side by side with the Standard Dropdown
    
    html.Div(children=[
        html.H2('Advanced Statistics:', style={'margin-right': '2em'}),
        dcc.Dropdown(id='advanced-dropdown',
                 options=[{'label':i, 'value':i} for i in advanced_cols],
                 value='WS',
                 placeholder="Select an Advanced Statistic here",
                 searchable=False
                ),
    
        dcc.Graph(id='advanced-chart', 
                       style={'display': 'flex','flex-direction':'column'})
    ],style={'width': '50%', 'display': 'inline-block'}),
])

#Creating the first callback to update the player-dropdown based on the values from the team-dropdown and player-dropdown
@app.callback([Output(component_id='player-dropdown', component_property='options'),
               Output(component_id='player-dropdown',component_property='value')],
              [Input(component_id='team-dropdown', component_property='value'),
               Input(component_id='player-dropdown', component_property='value')])


def get_player_options(team_value,player_value):
    #If the team_value is 'ALL', then the player dropdowns will include all players regardless of team
    if team_value=='ALL':
        return all_players_choices, player_value
    #Else, if a specific team is selected, then populate the player dropdown with all players in their active roster who have minutes this season (in unique_players)
    else:
        this_team=Team(team_value).get_roster()
        roster=this_team['Player']
        roster=roster[roster.isin(unique_players)==True]
        roster=sorted(roster.to_list())
        choices=[{'label':i, 'value':i} for i in roster]
        #If the pre-selected player shares the same team you entered, retain the original player value
        if Player(player_value).get_team()==team_value:
            player_result=player_value
        #Else, select the first active player on their alphabetically sorted roster
        else:
            player_result=roster[0]
            
        return choices, player_result

#Creating second callback to update charts when the standard-dropdown or advanced-dropdown value changes
@app.callback([Output(component_id='standard-chart', component_property='figure'),
               Output(component_id='advanced-chart', component_property='figure')],
              [Input(component_id='player-dropdown', component_property='value'),
               Input(component_id='standard-dropdown',component_property='value'),
               Input(component_id='advanced-dropdown',component_property='value')])

def get_stats(player_value,standard_stat_value,advanced_stat_value):
    global player_clusters
    #Instantiating the Player value to retrieve their top 3 cluster
    this_player=Player(player_value)
    
    if player_value not in player_clusters:
        players_list=this_player.get_top_3_cluster()
        players_list.insert(0,player_value)
        player_clusters[player_value]=players_list
    else:
        players_list=player_clusters[player_value]
    #Inserting the player name into their top 3 list so the player and their best peer group are all included
    #Creating the players_df_standard from the players_list
    players_df_standard=formatted_player_queries_standard[formatted_player_queries_standard['Player Name'].isin(players_list)]
    #Creating the fig_standard line chart
    fig1_standard=px.line(players_df_standard,x='Age',y=standard_stat_value,color='Player Name',category_orders={"Player Name":players_list})
    #Creating a scatter plot so can display players with only one season
    fig2_standard = px.scatter(players_df_standard,x='Age',y=standard_stat_value,color='Player Name',category_orders={"Player Name":players_list})
    
    #Creating combo line scatter plot
    fig_standard = go.Figure(data=fig1_standard.data + fig2_standard.data)
    
    #Removing duplicate traces
    traces=[]
    fig_standard.for_each_trace(
        lambda trace: trace.update(showlegend=False)  if trace.name in traces
        else traces.append(trace.name))
    
    fig_standard.update_layout(xaxis_title='Age',yaxis_title=standard_stat_value,title='{} {} Per Game'.format(player_value,standard_stat_value),template="presentation",title_x=0.5,font={'size':14},legend={'font':{'size':12},'orientation':'h','xanchor' : 'center', 'x' : 0.5, 'y': -.3})
    
    #Creating the players_df_advanced from the players_list
    players_df_advanced=formatted_player_queries_advanced[formatted_player_queries_advanced['Player Name'].isin(players_list)]
    #Creating the fig_advanced line chart from the players_df_advanced
    fig1_advanced=px.line(players_df_advanced,x='Age',y=advanced_stat_value,color='Player Name',category_orders={"Player Name":players_list})
    #Creating a scatter plot so can display players with only one season
    fig2_advanced = px.scatter(players_df_advanced,x='Age',y=advanced_stat_value,color='Player Name',category_orders={"Player Name":players_list})
    
    #Creating combo line scatter plot
    fig_advanced = go.Figure(data=fig1_advanced.data + fig2_advanced.data)
    
    #Removing duplicate traces
    traces=[]
    fig_advanced.for_each_trace(
        lambda trace: trace.update(showlegend=False)  if trace.name in traces
        else traces.append(trace.name))
    

    fig_advanced.update_layout(xaxis_title='Age',yaxis_title=advanced_stat_value,title='{} {}'.format(player_value,advanced_stat_value),template="presentation",title_x=0.5,font={'size':14},legend={'font':{'size':12},'orientation':'h','xanchor' : 'center', 'x' : 0.5, 'y': -.3})
    return fig_standard,fig_advanced


# Run the app
if __name__ == '__main__':
    app.run_server()

