# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 19:44:09 2023

@author:mark.pogson@liverpool.ac.uk

functions used by postproc.py
"""

import functions
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.stats as stats
import pickle
import string
matplotlib.rcParams.update({'font.size': 12})

class file():
    def read_parameters(param_names,folder=''): # this is no longer needed now pickle is being used, but kept in case method is useful in future for whatever reason
        parameters = {} # this will be a dictionary of {param_name: val, ...}
        with open(os.path.join(folder,'params.txt'),'r') as f:
            params = f.readlines()
        for param in params:
            name,val = param.split(' = ')
            if name in param_names:
                parameters[name] = float(val)
        return parameters # return parameters as a dictionary
    def unpickle_parameters(folder=''):
        with open(os.path.join(folder,'params.pkl'),'rb') as f:
            full_params = pickle.load(f)
        return full_params # return the dictionary of full_params (i.e. parameters that needed to be pickled as a dictionary of full lists)
    def increment_letter(letter,n): # take letter and return letter n along in the alphabet
        alphabet = list(string.ascii_lowercase) # used to increment subplot label for whatever reason
        letter_index = alphabet.index(letter) # index in alphabet for original subplot label
        return alphabet[letter_index+n]
    def capitalise_headers(df): # rename dataframe headers to be capitalised for use in figure labels
        headers0 = list(df)
        headers1 = [header.title() for header in headers0]
        rename_columns = dict(zip(headers0,headers1))
        df = df.rename(columns=rename_columns)
        return df

class collate():
    def to_array(a): # a is a list of arrays or dataframes for each Rep
        shapes = [np.shape(x) for x in a]
        if not all([s==shapes[0] for s in shapes]): # presume this is due to unequal number of Repetitions between trials - just duplicate existing Reps if one is integer multiple of other (unbiased way of using results in progress)
            nReps = [s[0] for s in shapes]
            nR = max(nReps)
            ratios = [nR/nr for nr in nReps]
            for i,ratio in enumerate(ratios):
                if ratio%1 != 0: # i.e. R not an integer multiple of r
                    print('failed - inconsistent number of Reps')
                    return
                else:
                    a[i] = np.Repeat(a[i],ratio,axis=0)
        return np.array(a) # (Rep,df dimensions); note that if multiple trials are present, this will be (trial,Rep,df,dimensions) - look at how files are read in
    def to_df(a): # a is list of dfs for each Rep
        dfs = [] # dfs is a list of dataframes for each Rep
        for r in range(len(a)):
            df = a[r]
            df['Rep'] = r # record Rep index
            dfs.append(df)
        dfs = pd.concat(dfs,ignore_index=True)
        return dfs
    def to_df_s(a,trials): # a is list of dfs for each Rep in each trial in trials
        dfss = []
        for l,trial in enumerate(trials):
            dfs = collate.to_df(a[l]) # properties across all Reps for current trial
            dfs['Trial'] = trial # record trial label
            dfss.append(dfs)
        dfss = pd.concat(dfss,ignore_index=True)
        dfss['Lrep'] = dfss['Trial'].astype(str)+'_'+dfss['Rep'].astype(str)
        return dfss

class time():
    def get_df_timeseries(timeseriess,trials,df_properties,variable_name,group_names,downsample_step):
        # input timeseries as an array of (Rep,time,agent) and df_properties as dataframe df[columns=agent,properties,Rep]
        # ouput dataframe as df[columns=time,agent,variable,Rep,group] (note that 'group' could in fact be any property listed in 'group_names')
        df_timeseries = []
        for l,timeseries in zip(trials,timeseriess):
            dfs = []
            for r,a in enumerate(timeseries): # a is (time,agent) for Rep r
                g = df_properties.loc[(df_properties['Trial']==l)&(df_properties['Rep']==r), ['Agent']+group_names] # agent id and group names
                df = time.array_to_df(a,g,variable_name=variable_name,columns=[],id_name='Agent',group_names=group_names,downsample_step=downsample_step) # this contains columns of agent, Rep, opinion time series
                dfs.append(df)
            dfs = pd.concat(dfs,ignore_index=True)
            dfs = dfs.drop(columns='Agent') # don't really need to keep the agent id - was only needed to join timeseries with agent properties; note that Rep and trial can be included in the group_names if these are wanted
            df_timeseries.append(dfs)
        df_timeseries = pd.concat(df_timeseries,ignore_index=True)
        #df_timeseries['Lrep'] = df_timeseries['Trial'].astype(str)+'_'+df_timeseries['Rep'].astype(str) # commented out to keep dataframe simpler
        return df_timeseries
    def array_to_df(a,g,variable_name,columns,id_name,group_names,downsample_step=10): # convert array a into a stacked dataframe with extra columns containing Rep r and group g, downsampled to keep every downsample_step row
        # turn array into dataframe, with headers showing agent index, i.e. 0,1,2,..
        if not columns: columns=range(a.shape[1])
        df = pd.DataFrame(a,columns=columns)
        # stack dataframe so agent time series are arranged vertically following each other, and rename columns accordingly
        df = df.stack().reset_index()
        rename_columns = dict(zip(list(df),['Time',id_name,variable_name]))
        df = df.rename(columns=rename_columns)
        # downsample dataframe to keep every downsample-th time; this avoids having an overly large dataframe
        if downsample_step==0: downsample_step=1 # ensure downsampling doesn't attempt to upsample (i.e. interpolate) instead
        filt = df['Time'].isin(range(0,a.shape[0],downsample_step))
        df = df[filt]
        # add columns to show which Rep and group the dataframe Represents, so it can later be concatenated with other Reps
        if len(g)>0: # i.e. groups are given, so match these according to agent id column
            for i in g['Agent'].unique(): # for each agent, get the agent id and group id
                for group_name in group_names:
                    group = g.loc[g['Agent']==i,group_name].item() # .item() needed in case group is a string, rather than a scalar, as this causes different behaviour in pandas for some reason
                    df.loc[df['Agent']==i,group_name] = group
        return df
    def get_snapshot(downsampled_t,timeseries,name):
        filt = (timeseries['Time']==downsampled_t)
        filtered = timeseries.loc[filt,:]
        seq = filtered.sort_values(by=['Rep','Agent'])
        vals = seq[name].values
        return vals
    def downsample(t,downsample_step): # get downsampled time, e.g. t=999 would be downsampled to t=800 if downsample_step=200)
        return int(t/downsample_step)*downsample_step

class analysis():
    def get_df_properties(properties):
        df_properties = []
        for i,prop in enumerate(properties):
            df = pd.DataFrame(prop)
            df['Rep'] = i
            df_properties.append(df)
        df_properties = pd.concat(df_properties,ignore_index=True)
        return df_properties  
    def get_neighbour_values(matrix, df_properties, property_name):
        all_neighbour_values = []
        for r in range(matrix.shape[0]): # each Rep
            values = df_properties.loc[df_properties['Rep']==r,property_name]
            neighbour_values = functions.analysis.get_neighbour_values(connection_matrix=matrix[r,:,:], values=values)
            all_neighbour_values.append(list(neighbour_values))
        all_neighbour_values = sum(all_neighbour_values, []) # flat list of neighbour values for each Rep
        return all_neighbour_values # this list is in the correct order to add to df_properties, i.e. fast cycle through agents, slow cycle through Reps
    def compare_distributions(distributions,test='ANOVA'): # distributions is a list of lists for each distribution being compared
        # one-way anova is usually better, even if normality can't be assumed https://stats.libretexts.org/Courses/Las_Positas_College/Math_40%3A_Statistics_and_Probability/12%3A_Nonparametric_Statistics/12.11%3A_KruskalWallis_Test
        if test=='ANOVA': # one-way anova gives the probability that the means are the same, assuming normality, though see above link for discussion about this
            stat,prob = stats.f_oneway(*distributions)
        elif test=='K-W': # kruskal is like a one-way ANOVA, but without assuming normality; it gives the probability that the medians are the same
            try:
                stat,prob = stats.kruskal(*distributions)
            except:
                stat,prob = [None,1]
        return stat,prob
    def get_groups_from_properties(df_properties,cluster='Rep'): # define property groups (e.g. small and large message rate) based on property values for each given group (e.g. Rep, trial)
        def group_by_median(isg,prop):
            mv = df_properties.loc[isg,prop].median()
            groups = df_properties.loc[isg,prop].apply(
                    lambda x: 0 if x<mv else 1)
            return groups.to_list()
        gs = sorted(df_properties[cluster].unique())
        for i,g in enumerate(gs):
            isg = (df_properties[cluster]==g) # filter for current cluster, e.g. Rep
            for prop in ['Sent Count','Received Count','Final Opinion','Roughness']:#,'Initial Opinion','Message Rate']: # commented out initial opinion and message rate as these were output from the model script, but could define differently here if wanted
                if prop in df_properties.columns: # possible certain properties weren't output from the simulation
                    groups = group_by_median(isg,prop)
                    df_properties.loc[isg,prop+' Group'] = groups
        return df_properties
    def get_cluster_vals(df_properties,cluster='Lrep'): # get single values from each specified cluster (e.g. Lrep, Rep, trial) to correlate
        df_sing = pd.DataFrame() # this will be a dataframe of single values from each Rep, e.g. initial opinion of agent with highest message rate, mean final opinion
        gs = sorted(df_properties[cluster].unique())
        for i,g in enumerate(gs): # go through unique cluster values, e.g. each trial, and use these to add rows to df_sing
            # record cluster value, e.g. trial label, Rep number
            df_sing.loc[i,cluster] = g
            # create filter for the current cluster in df_properties
            isg = (df_properties[cluster]==g) # filter for current cluster, e.g. Lrep
            # find MAA index for the current cluster in df_properties
            MAA = df_properties.loc[isg,'Message Rate'].idxmax() # row index for agent with largest message rate in current cluster, e.g. Rep
            MAA_id = df_properties.loc[MAA,'Agent'] # MAA agent index in Lrep (distinct from its row index)
            # record in df_sing groups for the MAA
            for prop in df_properties.columns:
                if 'Group' in prop:
                    df_sing.loc[i,prop+' MAA'] = df_properties.loc[MAA,prop]
            # record in df_sing MAA initial opinion, and cluster mean final opinion
            df_sing.loc[i,'Initial Opinion MAA'] = df_properties.loc[MAA,'Initial Opinion'] # initial opinion of MAA
            df_sing.loc[i,'Mean Final Opinion'] = df_properties.loc[isg,'Final Opinion'].mean() # mean final opinion of agents in current cluster, e.g. Rep
            # record in df_properties MAA initial opinion
            df_properties.loc[isg,'Initial Opinion group MAA'] = df_properties.loc[MAA,'Opinion Group'] # record this for all agents in current cluster, e.g. Lrep
            df_properties.loc[isg,'MAA'] = df_properties.loc[isg,'Agent'].apply(lambda x:True if x==MAA_id else False) # True if agent is MAA in cluster, else False
        return df_sing, df_properties
    def get_entities(df_properties,comparison_entity,all_trials): # define entities and hues to compare (do this before relabelling trial-group)
        hue = comparison_entity    
        entities = list(df_properties[hue].unique())
        ne = len(entities)
        if hue=='Trial' and all([entity in all_trials for entity in entities]): # i.e. all trials are in all_trials
            all_entities = all_trials # make sure the colours are consistent for all trials, even ones not present in the comparison
        else:
            all_entities = entities # make sure colours are consistent for the trials present
        all_colours = sns.color_palette('colorblind',len(all_entities)) # can't just pass the palette name, as want full palette even if not all trials are present
        all_palette = dict(zip(all_entities,all_colours))
        palette = [all_palette[x] for x in entities] # pick out palette for current entities being compared
        return entities,ne,hue,palette,all_palette

class tension():
    def get_df_tensions(tensionsss,trials,columns,group_types,downsample_step=10): # take list of tension arrays for each Rep and turn into stacked dataframe
        df_tensions = []
        for l,tensionss in zip(trials,tensionsss):
            df_tensions.append([])
            for i,tensions in enumerate(tensionss):
                dft = []
                for r,tension in enumerate(tensions): # tensions from each Rep
                    df = time.array_to_df(tension,g=[],variable_name='Tension',columns=columns,id_name='Tension Form',group_name='',downsample_step=downsample_step)
                    df['Rep'] = r
                    dft.append(df)
                dft = pd.concat(dft,ignore_index=True)
                dft['Group Type'] = group_types[i]
                df_tensions[-1].append(dft)
            df_tensions[-1] = pd.concat(df_tensions[-1],ignore_index=True)
            df_tensions[-1]['Trial'] = l
        df_tensions = pd.concat(df_tensions,ignore_index=True)
        df_tensions['Lrep'] = df_tensions['Trial'].astype(str)+'_'+df_tensions['Rep'].astype(str)
        return df_tensions

class plot():
    def plot_timeseries(df, variable_name, title='', hue=None, style=None, palette=None, p0=True, mlo=False, estimator='mean', n_boot=1000, seed=None, spl='',folder=''): # df is a dataframe containing time, agent id, variable, Rep and specified group
        # order hue and style sequentially if present, and make strings if integer
        by = [hue,style] if style else [hue]
        if hue:
            df = df.sort_values(by=by)
            for col in by:
                if functions.analysis.isint(df[col]): # use my isint function rather than pandas dtype, as it covers any kind of int
                    df[col]=df[col].apply(lambda x:str(int(x))) # make hue column a string of the integer value (otherwise seaborn wants to write 0.0, 1.0, etc)
        # plot
        fig = plt.figure(figsize=[6,4]); ax = fig.add_subplot()
        sns.lineplot(data=df, x='Time', y=variable_name, hue=hue, style=style, palette=palette, estimator=estimator, n_boot=n_boot, seed=seed, ax=ax)
        # add dashed line for y=0
        if p0: ax.plot([df['Time'].min(),df['Time'].max()],[0,0],'k:') # plot dashed line for y=0
        # move legend
        if (hue or style) and mlo: sns.move_legend(ax, bbox_to_anchor=(1,1), loc='upper left')
        # add title, subplot label and save
        if title: plt.title(title)
        if spl: ax = functions.plot.add_subplot_label(ax,spl)
        plt.savefig(os.path.join(folder,'timeseries_'+functions.file.underscore(variable_name)+'_by_'+functions.file.underscore(' '.join(by))+'.png'), dpi=200, bbox_inches='tight')
        plt.close()
        return
    def get_unique_hues(df,hue):
        unique_hues = df[hue].unique()
        palette = sns.color_palette('colorblind',len(unique_hues))
        return unique_hues, palette
    def plot_distribution(df, variable_name, hue=None, palette=None, ptype='hist', midlines=False, binrange=None, spl='', folder=''):
        # get hue-level information and use colours to Represent these
        if hue:
            unique_hues, pal = plot.get_unique_hues(df,hue) # set actual palette so can be used outside seaborn plotting function, e.g. for annotation text colour
            if not palette: # don't want to Replace given palette
                palette = pal
        else:
            unique_hues = []; palette = []
        # plot distribution
        fig = plt.figure(figsize=[6,4]); ax = fig.add_subplot()
        if ptype=='hist':
            sns.histplot(df, x=variable_name, hue=hue, palette=palette, binrange=binrange, stat='percent', multiple='dodge', common_norm=False, ax=ax)
        elif ptype=='violin':
            sns.violinplot(df, y=variable_name, x=hue, palette=palette, ax=ax)
        else:
            sns.boxplot(df, y=variable_name, x=hue, palette=palette, ax=ax)
        means = []; medians = []
        for i,h in enumerate(unique_hues):
            mean = df.loc[df[hue]==h,variable_name].mean()
            median = df.loc[df[hue]==h,variable_name].median()
            means.append(mean); medians.append(median)
            for j,(name,stat) in enumerate(zip(['Median','Mean'],[median,mean])):
                ax.annotate(name+' = %.2f'%stat, xy=(0.35*i, 1.02+0.06*j), xycoords='axes fraction',color=palette[i])
        # show 0 in distribution
        if ptype=='hist':
            pass
            ys = ax.get_ylim()
            yrange = ys[1]-ys[0]
            if 0<df[variable_name].max() and 0>df[variable_name].min(): # i.e. 0 is within data range
                ax.plot([0,0],ys,color='black',ls='dotted') # vertical line for 0
            if midlines: # add vertical lines for the mean and median of each hue
                for i,(mean,media) in enumerate(zip(means,medians)):
                    ax.plot([mean,mean], ys, ls=(0,(5,10)), c=palette[i])
                    ax.plot([median,median], ys, ls='dashdot', c=palette[i])
            ax.set_ylim(ys)
        else:
            xs = ax.get_xlim()
            ax.plot(xs,[0,0],'k:') # horizontal line for 0
        # get separate distributions for each hue and compare according to test
        if len(unique_hues)>1: # i.e. have different groups to compare distributions for
            if len(unique_hues)<3:
                xan = 0.7; yan = 1.02
            else:
                xan = 0.5; yan = 0.84
            distributions = [df.loc[df[hue]==h,variable_name].values for h in unique_hues]
            for i,test in enumerate(['K-W','ANOVA']):
                stat,prob = analysis.compare_distributions(distributions,test)
                ax.annotate(test+' $p$ = %.2f'%prob, (xan, yan+0.06*i), xycoords='axes fraction')
        # add subplot label and save
        if spl: ax = functions.plot.add_subplot_label(ax,spl)
        plt.savefig(os.path.join(folder,functions.file.underscore('histogram_'+variable_name+'_'+hue+'_'+'_'.join([str(x) for x in unique_hues])+'_'+'.png')), dpi=200, bbox_inches='tight')
        plt.close()
        return
    def plot_joint(df, variable_name1, variable_name2, hue=None, folder=''):
        # currently can't use spl with this - would need to fiddle about with JointGrid and extract suitable axes object
        ax = sns.jointplot(x=df[variable_name1], y=df[variable_name2], hue=hue, kind='kde', fill=True)
        ax.set_axis_labels(variable_name1, variable_name2)
        plt.savefig(os.path.join(folder,'jointplot_'+functions.file.underscore(variable_name1)+'_'+functions.file.underscore(variable_name2)+'.png'), dpi=200, bbox_inches='tight')
        plt.close()
        return
    def plot_correlation(df, x='', y='', hue=None, palette=None, spl='', folder=''):
        # define colour to use for markers and annotations
        if hue:
            unique_hues, palette = plot.get_unique_hues(df,hue) # set actual palette so can be used outside seaborn plotting function
        else:
            unique_hues = ['all']; palette = ['black'] # seaborn will ignore this, which is fine, but it's still used for the annotation color
        # plot scatter and label axes
        g = sns.lmplot(df, x=x, y=y, hue=hue, palette=palette)
        ax = g.axes[0,0]
        ax.set_xlabel(x); ax.set_ylabel(y)
        # plot horizontal and vertical 0 lines
        x0,x1 = ax.get_xlim(); y0,y1 = ax.get_ylim()
        ax.plot([x0,x1],[0,0],'k:') # plot dashed line for y=0
        ax.plot([0,0],[y0,y1],'k:') # plot dashed line for x=0
        ax.set_xlim([x0,x1]); ax.set_ylim([y0,y1])
        # add annotation(s) for correlation between x and y (distinguished by hue if relevant)
        i = 0
        for j,h in enumerate(sorted(unique_hues)):
            if hue:
                ish = (df[hue]==h)
                prefix = hue+' '+str(h)+' '
            else:
                ish = df.iloc[:,0].apply(lambda _: True) # select all rows in a way consistent with above filter
                prefix = ''
            if ish.sum()>0:
                lr = stats.linregress(x=df.loc[ish,x], y=df.loc[ish,y])
                ax.annotate(prefix+'$R$ = %.2f'%lr.rvalue+', $p$ = %.2f'%lr.pvalue, (0.05,0.9-0.1*i), xycoords='axes fraction',color=palette[j])
                i+=1
        # add subplot label and save
        if spl: ax = functions.plot.add_subplot_label(ax,spl)
        plt.savefig(os.path.join(folder,'scatterplot_'+functions.file.underscore(x)+'_'+functions.file.underscore(y)+'.png'), dpi=200, bbox_inches='tight')
        plt.close()
        return
    
if __name__=='__main__':
    gt = False # True to test get_df_group_timeseries
    ge = False # True to test get_snapshot
    gd = False # True to test plot_distribution
    gc = False # True to test plot_correlation
    gp = False # True to test plot_timeseries
    gl = False # True to test increment_letter
    if gt:
        variable_name = 'opinion'
        group_name = 'group'
        # define array of timeseries values with dimensions (Rep,time,agent), with values containing (Rep.time,agent,group) using 4 s.f. to Represent the corresponding index values
        # this example has 3 time point, 3 agents, 2 Reps and 2 groups (agents 2 and 1 change from group 1 and 0 in the 2 Reps)
        Rep0 = [[0.000,0.010,0.021],[0.100,0.110,0.121],[0.200,0.210,0.221]]
        Rep1 = [[1.000,1.011,1.020],[1.100,1.111,1.120],[1.200,1.211,1.220]]
        timeseries = np.array([Rep0,Rep1])
        # define array for each group giving Boolean for being in group, listed for each group, i.e. [group,(Rep,agent)]
        Rep0 = [[True,True,False],[True,False,True]]
        Rep1 = [[False,False,True],[False,True,False]]
        isgroups = [np.array(Rep0),np.array(Rep1)]
        # get timeseries as dataframe including column for group
        df = time.get_df_timeseries([timeseries],isgroups,variable_name=variable_name,group_name=group_name,downsample_step=1)
    if ge:
        variable_name = 'opinion'
        # define dataframe df[cols=time,agent,opinion,Rep,group] with opinion values containing (time,agent,Rep,group) using 4 s.f. to Represent the corresponding index values
        # this example has 3 time point, 3 agents, 2 Reps and 2 groups (agents 2 and 1 change from group 1 and 0 in the 2 Reps)
        df = pd.DataFrame([[0,0,0.000,0,0],[0,1,0.100,0,0],[0,2,0.201,0,1],[1,0,1.000,0,0],[1,1,1.100,0,0],[1,2,1.201,0,1],[2,0,2.000,0,0],[2,1,2.100,0,0],[2,2,2.201,0,1],[0,0,0.010,1,0],[0,1,0.111,1,0],[0,2,0.210,1,1],[1,0,1.010,1,0],[1,1,1.111,1,0],[1,2,1.210,1,1],[2,0,2.010,1,0],[2,1,2.111,1,0],[2,2,2.210,1,1]],columns=['Time','Agent',variable_name,'Rep','group'])
        t=1
        snapshot_values = time.get_snapshot(t,df,variable_name)
    if gd:
        df = pd.DataFrame()
        df['vals']=[234,345,345,45,324]
        df['group']=[0,0,1,1,1]
        plot.plot_distribution(df, 'vals', 'group')
    if gc:
        df = pd.DataFrame()
        xs = np.linspace(1,30,30)
        hs = [np.random.randint(3) for _ in xs]
        ys = [x+h+np.random.rand() for x,h in zip(xs,hs)]
        df['vals1']=xs
        df['vals2']=ys
        df['group']=hs
        plot.plot_correlation(df, x='vals1', y='vals2', hue='group')
    if gp:
        df = pd.DataFrame({'Time':[0,1,2,3,0,1,2,3,0,1,2,3],'vals':[0,1,2,3,0.5,1.5,2.5,3.5,4,5,6,7],'form':['a','a','a','a','a','a','a','a','b','b','b','b'],'type':['p','p','q','q','p','p','p','p','r','r','s','s']})
        plot.plot_timeseries(df,variable_name='vals',hue='form',style='type',palette='colorblind',p0=False)
    if gl:
        print(file.increment_letter('b',3))