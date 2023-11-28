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
    def to_array(a): # a is a list of arrays or dataframes for each rep
        shapes = [np.shape(x) for x in a]
        if not all([s==shapes[0] for s in shapes]): # presume this is due to unequal number of repetitions between trials - just duplicate existing reps if one is integer multiple of other (unbiased way of using results in progress)
            nreps = [s[0] for s in shapes]
            nR = max(nreps)
            ratios = [nR/nr for nr in nreps]
            for i,ratio in enumerate(ratios):
                if ratio%1 != 0: # i.e. R not an integer multiple of r
                    print('failed - inconsistent number of reps')
                    return
                else:
                    a[i] = np.repeat(a[i],ratio,axis=0)
        return np.array(a) # (rep,df dimensions); note that if multiple trials are present, this will be (trial,rep,df,dimensions) - look at how files are read in
    def to_df(a): # a is list of dfs for each rep
        dfs = [] # dfs is a list of dataframes for each rep
        for r in range(len(a)):
            df = a[r]
            df['Rep'] = r # record rep index
            df = file.capitalise_headers(df) # do this here to ensure concatenated dfs have consistent headers (capitalisation wasn't always used in older results)
            dfs.append(df)
        dfs = pd.concat(dfs,ignore_index=True)
        return dfs
    def to_df_s(a,trials): # a is list of dfs for each rep in each trial in trials
        dfss = []
        for l,trial in enumerate(trials):
            dfs = collate.to_df(a[l]) # properties across all reps for current trial
            dfs['Trial'] = trial # record trial label
            dfss.append(dfs)
        dfss = pd.concat(dfss,ignore_index=True)
        dfss['Lrep'] = dfss['Trial'].astype(str)+'_'+dfss['Rep'].astype(str)
        return dfss

class time():
    def get_df_timeseries(timeseriess,trials,df_properties,variable_name,group_names,downsample_step):
        # input timeseries as an array of (rep,time,agent) and df_properties as dataframe df[columns=agent,properties,rep]
        # ouput dataframe as df[columns=time,agent,variable,rep,group] (note that 'group' could in fact be any property listed in 'group_names')
        df_timeseries = []
        for l,timeseries in zip(trials,timeseriess):
            dfs = []
            for r,a in enumerate(timeseries): # a is (time,agent) for rep r
                g = df_properties.loc[(df_properties['Trial']==l)&(df_properties['Rep']==r), ['Agent']+group_names] # agent id and group names
                df = time.array_to_df(a,g,variable_name=variable_name,columns=[],id_name='Agent',group_names=group_names,downsample_step=downsample_step) # this contains columns of agent, rep, opinion time series
                dfs.append(df)
            dfs = pd.concat(dfs,ignore_index=True)
            dfs = dfs.drop(columns='Agent') # don't really need to keep the agent id - was only needed to join timeseries with agent properties; note that rep and trial can be included in the group_names if these are wanted
            df_timeseries.append(dfs)
        df_timeseries = pd.concat(df_timeseries,ignore_index=True)
        #df_timeseries['Lrep'] = df_timeseries['Trial'].astype(str)+'_'+df_timeseries['Rep'].astype(str) # commented out to keep dataframe simpler
        return df_timeseries
    def array_to_df(a,g,variable_name,columns,id_name,group_names,downsample_step=10): # convert array a into a stacked dataframe with extra columns containing rep r and group g, downsampled to keep every downsample_step row
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
        # add columns to show which rep and group the dataframe represents, so it can later be concatenated with other reps
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
        for r in range(matrix.shape[0]): # each rep
            values = df_properties.loc[df_properties['Rep']==r,property_name]
            neighbour_values = functions.analysis.get_neighbour_values(connection_matrix=matrix[r,:,:], values=values)
            all_neighbour_values.append(list(neighbour_values))
        all_neighbour_values = sum(all_neighbour_values, []) # flat list of neighbour values for each rep
        return all_neighbour_values # this list is in the correct order to add to df_properties, i.e. fast cycle through agents, slow cycle through reps
    def compare_distributions(distributions,test='ANOVA'): # distributions is a list of lists for each distribution being compared
        # one-way anova is usually better, even if normality can't be assumed https://stats.libretexts.org/Courses/Las_Positas_College/Math_40%3A_Statistics_and_Probability/12%3A_Nonparametric_Statistics/12.11%3A_KruskalWallis_Test
        if test=='ANOVA': # one-way anova gives the probability that the means are the same, assuming normality, though see above link for discussion about this
            stat,prob = stats.f_oneway(*distributions)
        elif test=='K-W': # kruskal is like a one-way ANOVA, but without assuming normality; it gives the probability that the medians are the same
            try:
                stat,prob = stats.kruskal(*distributions)
            except:
                stat,prob = [None,1]
        elif test=='ES': # Cohen's d effect size gives difference in means as fraction of overall standard deviation
            stat = np.nan # dummy value
            prob = np.nan # placeholder values
            if len(distributions)==2: # can only calculate Cohen's d on 2 distributions, where control group here is distributions[1]
                n0 = len(distributions[0])
                n1 = len(distributions[1])
                s = np.sqrt(((n0-1)*np.std(distributions[0],ddof=1) ** 2 + (n1-1)*np.std(distributions[1],ddof=1) ** 2) / (n0+n1-2))    
                prob = (np.mean(distributions[1])-np.mean(distributions[0]))/s # https://en.wikipedia.org/wiki/Effect_size#Cohen's_d
        return stat,prob
    def get_groups_from_properties(df_properties,cluster='Rep'): # define property groups (e.g. small and large message rate) based on property values for each given group (e.g. rep, trial)
        def group_by_median(isg,prop):
            mv = df_properties.loc[isg,prop].median()
            groups = df_properties.loc[isg,prop].apply(
                    lambda x: 0 if x<mv else 1)
            return groups.to_list()
        gs = sorted(df_properties[cluster].unique())
        for i,g in enumerate(gs):
            isg = (df_properties[cluster]==g) # filter for current cluster, e.g. rep
            for prop in ['Sent Count','Received Count','Final Opinion','Roughness']:#,'Initial Opinion','Message Rate']: # commented out initial opinion and message rate as these were output from the model script, but could define differently here if wanted
                if prop in df_properties.columns: # possible certain properties weren't output from the simulation
                    groups = group_by_median(isg,prop)
                    df_properties.loc[isg,prop+' Group'] = groups
        return df_properties
    def get_cluster_vals(df_properties,cluster='Lrep'): # get single values from each specified cluster (e.g. Lrep, rep, trial) to correlate
        df_sing = pd.DataFrame() # this will be a dataframe of single values from each rep, e.g. initial opinion of agent with highest message rate, mean final opinion
        gs = sorted(df_properties[cluster].unique())
        for i,g in enumerate(gs): # go through unique cluster values, e.g. each trial, and use these to add rows to df_sing
            # record cluster value, e.g. trial label, rep number
            df_sing.loc[i,cluster] = g
            # create filter for the current cluster in df_properties
            isg = (df_properties[cluster]==g) # filter for current cluster, e.g. Lrep
            # find MAA index for the current cluster in df_properties
            MAA = df_properties.loc[isg,'Message Rate'].idxmax() # row index for agent with largest message rate in current cluster, e.g. rep
            MAA_id = df_properties.loc[MAA,'Agent'] # MAA agent index in Lrep (distinct from its row index)
            # record in df_sing groups for the MAA
            for prop in df_properties.columns:
                if 'Group' in prop:
                    df_sing.loc[i,prop+' MAA'] = df_properties.loc[MAA,prop]
            # record in df_sing MAA initial opinion, and cluster mean final opinion
            df_sing.loc[i,'Initial Opinion MAA'] = df_properties.loc[MAA,'Initial Opinion'] # initial opinion of MAA
            df_sing.loc[i,'Mean Final Opinion'] = df_properties.loc[isg,'Final Opinion'].mean() # mean final opinion of agents in current cluster, e.g. rep
            # record in df_properties MAA initial opinion
            df_properties.loc[isg,'Initial Opinion Group MAA'] = df_properties.loc[MAA,'Opinion Group'] # record this for all agents in current cluster, e.g. Lrep
            df_properties.loc[isg,'MAA'] = df_properties.loc[isg,'Agent'].apply(lambda x:True if x==MAA_id else False) # True if agent is MAA in cluster, else False
        return df_sing, df_properties
    def get_entities(df_properties,comparison_entity,all_trials): # define entities and corresponding colours to compare (do this before relabelling trial-group)     
        entities = list(df_properties[comparison_entity].unique())
        ne = len(entities)
        if comparison_entity=='Trial' and all([entity in all_trials for entity in entities]): # i.e. all trials are in all_trials
            all_entities = all_trials # make sure the colours are consistent for all trials, even ones not present in the comparison
        else:
            all_entities = entities # just make sure colours are consistent for the trials present here
        all_colours = sns.color_palette('colorblind',len(all_entities)) # can't just pass the palette name, as want full palette even if not all trials are present
        all_palette = dict(zip(all_entities,all_colours))
        palette = [all_palette[x] for x in entities] # pick out palette for current entities being compared
        return entities,ne,palette,all_palette
    def make_filter(df,col,val,condition=True):
        if condition: # i.e. want filter
            filt = (df[col]==val)
        else: # i.e. want no filter, so return Series of True with correct indexing
            filt = (df.iloc[:,0].apply(lambda _: True))
        return filt
    def get_ci(vals,kind='mean',ci=0.95):
        if kind=='mean': # calculate confidence interval around mean
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
            # df is degrees of freedom, loc is the mean of vals, and scale is the standard error of the mean of vals, i.e. standard deviation
            # note that the scipy documentation is incorrect when it states the ci is about the median - it is about the mean, as you'd expect from t distribution; testing shows this to be the case
            ci = stats.t.interval(confidence=ci, df=len(vals)-1, loc=np.mean(vals), scale=stats.sem(vals))
        elif kind=='median': # calculate confidence interval around median
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.median_cihs.html
            ci = stats.mstats.median_cihs(vals, alpha=1-ci)
        return ci

class tension():
    def get_df_tensions(tensionsss,trials,columns,group_types,downsample_step=10): # take list of tension arrays for each rep and turn into stacked dataframe
        df_tensions = []
        for l,tensionss in zip(trials,tensionsss):
            df_tensions.append([])
            for i,tensions in enumerate(tensionss):
                dft = []
                for r,tension in enumerate(tensions): # tensions from each rep
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
    def plot_timeseries(df, variable_name, title='', hue=None, hue_order=None, style=None, palette=None, facet=None, p0=True, mlo=False, estimator='mean', n_boot=1000, seed=None, spl='',folder=''): # df is a dataframe containing time, agent id, variable, rep and specified group
        # order hue and style sequentially if present, and make strings if integer
        by = [hue,style] if style else [hue]
        if hue and hue_order!=None:
            df = df.sort_values(by=by)
            for col in by:
                if functions.analysis.isint(df[col]): # use my isint function rather than pandas dtype, as it covers any kind of int
                    df[col]=df[col].apply(lambda x:str(int(x))) # make hue column a string of the integer value (otherwise seaborn wants to write 0.0, 1.0, etc)
        # plot
        if facet!=None: # this won't include a subplot label, as only likely to be used as standalone figure
            g = sns.FacetGrid(data=df,col=facet,col_wrap=2,hue=hue,palette=palette) # I don't think there's a way to include style here (hue_kw doesn't quite do it - you can change style, but only in line with the hue)
            g.map_dataframe(sns.lineplot,x='Time',y=variable_name) # putting style here does nothing, and see above line for using it in facetgrid itself
            nh = len(df[hue].unique())
            if nh>1:
                g.add_legend()
                sns.move_legend(g, loc='lower center',bbox_to_anchor=(0.5, 1), ncol=nh)
            axes = g.axes.flatten()
            for ax in axes:
                title = ax.get_title()
                di = title.find(' = ')
                newtitle = title[di+3:].replace('Minority Group Size (Fraction)','Minority Fraction').replace('Opposing Pulse Time Delay','Opposing Delay') # simplify title for minority group
                ax.set_title(newtitle) # removing 'description = '
        else:
            fig = plt.figure(figsize=[6,4]); ax = fig.add_subplot()
            sns.lineplot(data=df, x='Time', y=variable_name, hue=hue, style=style, palette=palette, estimator=estimator, n_boot=n_boot, seed=seed, ax=ax)
            # add dashed line for y=0
            if p0: ax.plot([df['Time'].min(),df['Time'].max()],[0,0],'k:') # plot dashed line for y=0
            # move legend
            if (hue or style) and mlo: sns.move_legend(ax, bbox_to_anchor=(1,1), loc='upper left')
            # add title, subplot label and save
            if title: plt.title(title)
            if spl: ax = functions.plot.add_subplot_label(ax,spl)
        plt.savefig(os.path.join(folder,'timeseries_'+functions.file.underscore(variable_name)+'_by_'+functions.file.underscore('_'.join(by))+'_'+spl+'.png'), dpi=200, bbox_inches='tight')
        plt.close()
        return
    def get_unique_hues(df,hue):
        unique_hues = df[hue].unique()
        palette = sns.color_palette('colorblind',len(unique_hues))
        return unique_hues, palette
    def plot_distribution(df, variable_name, hue=None, palette=None, ptype='hist', midlines=False, bins='auto', binrange=None, show_ES=False, spl='', suffix='', folder=''):
        # get hue-level information and use colours to represent these
        if hue: # need to do this in a longwinded way to deal with annotation colours in the same way as seaborn semantic
            unique_hues, pal = plot.get_unique_hues(df,hue) # set actual palette so can be used outside seaborn plotting function, e.g. for annotation text colour
            if not palette: # don't want to replace given palette
                palette = pal
        else:
            unique_hues = []; palette = []
        # plot distribution
        fig = plt.figure(figsize=[6,4]); ax = fig.add_subplot()
        if ptype=='hist':
            sns.histplot(df, x=variable_name, hue=hue, palette=palette, bins=bins, binrange=binrange, stat='percent', multiple='dodge', common_norm=False, ax=ax)
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
            #yrange = ys[1]-ys[0]
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
            distributions = [df.loc[df[hue]==h,variable_name].dropna().values for h in unique_hues]
            for i,test in enumerate(['K-W','ANOVA']):
                stat,prob = analysis.compare_distributions(distributions,test)
                ax.annotate(test+' $p$ = %.2f'%prob, (xan, yan+0.06*i), xycoords='axes fraction')
        # add effect size annotation (this should be used with caution, as it makes assumptions about normality and variance that may not be suitable)
        if show_ES:
            test = 'ES'
            _,d = analysis.compare_distributions(distributions,test)
            if not np.isnan(d): ax.annotate(test+' $d$ = %.2f'%d, (0.4, yan-0.1), xycoords='axes fraction')
            # add subplot label and save
        if spl: ax = functions.plot.add_subplot_label(ax,spl)
        plt.savefig(os.path.join(folder,functions.file.underscore('histogram_'+variable_name+'_'+hue+'_'+'_'.join([str(x) for x in unique_hues])+'_'+suffix+'.png')), dpi=200, bbox_inches='tight')
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
    def plot_sensitivity(df, x='', y='', hue='Trial', linestyle='Average', markerstyle='Test $p$', ym='$p$', ci='ci', base_num = None, legend = 'auto', palette = 'colorblind', spl='', folder=''):
        df[x] = df[x].astype(float) # ensures sensitivity numbers are floats, as previously had string present for comparison trial
        # create twin axes
        fig = plt.figure(figsize=[6,4]); ax = fig.add_subplot(); ax2 = ax.twinx()
        # determine whether/where to include legend (can control this with the legend argument with suffix to auto)
        leg1 = 'auto' if legend=='auto' or legend=='auto1' else None
        leg2 = 'auto' if legend=='auto' or legend=='auto2' else None
        # plot averages on first axes, and show test probabilitities ps on the twin axes using markers
        sns.lineplot(data=df, x=x, y=y, hue=hue, style=linestyle, palette=palette, legend=leg1, ax=ax)
        sns.scatterplot(data=df.dropna(), x=x, y=ym, style=markerstyle, color='k', s=75, legend=leg2, ax=ax2)
        # add error bands for lineplot (seaborn doesn't offer direct way of doing this with a summary dataframe)
        domn = True if 'Mean' in df[linestyle].to_list() else False
        domd = True if 'Median' in df[linestyle].to_list() else False
        if domn and domd:
            colours = sns.color_palette(palette)
            linestyles = ['-','--']
            if legend=='simple':
                markerstyles = ['o','X'] # only need these if making full simple legend
                markerlabels = ['ANOVA','K-W']
            alphas = [0.2,0.1]
            trials = df[hue].unique()
            is_mean = df[linestyle]=='Mean' # only 'Mean' and 'Median' present, so only need filter for one
            if legend=='simple': is_ANOVA = df[markerstyle]=='ANOVA' # " " ANOVA and K-W
            for i,trial in enumerate(trials):
                is_trial = (df[hue]==trial)
                xs = df.loc[is_trial & is_mean,x].values # can use either mean or median here, they both have the same x values
                if legend=='simple' or legend=='simple1':
                    ys = df.loc[is_trial & is_mean,y].values # used for dummy curve to make simple legend
                    if legend!='simple1':
                        ps = df.loc[is_trial & is_ANOVA,ym].values
                for j,kind in enumerate(['Mean','Median']):
                    if kind=='Mean':
                        cis = df.loc[is_trial&is_mean,ci].values # ci is the column header in df which gives the pre-calculated confidence intervals
                    else:
                        cis = df.loc[is_trial&~is_mean,ci].values
                    lows = [v[0] for v in cis]
                    highs = [v[1] for v in cis]
                    if  legend=='simple' or legend=='simple1': # make dummy band for legend labels
                        if j==0: # only need to do hue labels once
                            ax.fill_between([xs[0]], [ys[0]], [ys[0]], color=colours[i], label=trial)
                    if legend=='simple' or legend=='simple2': # i.e. want linestle and marker shown with ax2
                        if i==0: # line and marker style is consistent across hue, so only do once for each style
                            ax2.plot([xs[0]], [ys[0]], c='k', ls=linestyles[j], label=kind)
                            ax2.plot([xs[0]], [ys[0]], c='k', ls='', marker=markerstyles[j], label=markerlabels[j])
                    ax.fill_between(xs, lows, highs, linestyle=linestyles[j], color=colours[i], alpha=alphas[j]) # colour matches trial, other semantics match average kind
        # set limits for ax2 y-axis
        pmax = df[ym].max() if df[ym].max()>0.05 else 0.05 # i.e. avoiding ylims both being close to 0, which causes matplotlib to adjust them in an unhelpful way
        ax2.set_ylim([0,pmax]) # i.e. only show y-axis for p values up to maximum present
        # make base x ticklabel bold and deal with any potential overlap
        xticks = sorted(df[x].unique().tolist())
        xrange = xticks[-1]-xticks[0]
        ax.set_xticks(xticks) # make the ticklabels match the senstivity numbers
        labels = ax.get_xticklabels()
        #labels = ['%.2f'%np.round(float(l.get_text()),2) for l in labels] # make label 2 d.p.
        #labels = [plt.text(xt,0,'%.2f'%np.round(xt)) for xt in xticks] # make label 2 d.p.
        if base_num!=None:
            ind = xticks.index(base_num) # index of base num in xticks
            labels[ind].set_weight('bold')
        for i in range(len(xticks)-1):
            j = i+1
            #if xticks[j]-xticks[i]<xrange/10:
            #    labels[i].set_rotation(15) #labels[i].set_ha('right') # move ith ticklabel to the left of the tick
        ax.set_xticks(xticks,labels=labels,rotation=35,ha='right')
        # move legend(s) if present
        if legend=='simple' or legend=='simple1': # i.e. just use line colour from error band labels, so can describe style and marker in caption for less cluttered figure
            ax.legend(title=hue, bbox_to_anchor=(0,1), loc='lower left', ncol=2)
        if legend=='simple' or legend=='simple2':
            ax2.legend(bbox_to_anchor=(1,1), loc='lower right', ncol=2)
        if leg1!=None: # by definition, leg will be None if legend is any form of simple
            sns.move_legend(ax, bbox_to_anchor=(-0.04,1), loc='lower left', ncol=2)
        if leg2!=None:
            sns.move_legend(ax2, bbox_to_anchor=(1.1,1), loc='lower right', ncol=2)
        # add subplot label and save
        if spl: ax = functions.plot.add_subplot_label(ax,spl)
        plt.savefig(os.path.join(folder,'sensitivity_plot_'+spl+'.png'), dpi=200, bbox_inches='tight')
        plt.close()
        return
    
if __name__=='__main__':
    gt = False # True to test get_df_group_timeseries, and also plot it
    ge = False # True to test get_snapshot
    gd = False # True to test plot_distribution
    gc = False # True to test plot_correlation
    gp = False # True to test plot_timeseries
    gl = False # True to test increment_letter
    gs = False # True to test get_ci and plot_sensitivity
    if gt:
        variable_name = 'opinion'
        group_names = ['group','Trial']
        # define array of timeseries values with dimensions (rep,time,agent), with values containing (rep,time,agent,group) using 4 s.f. to represent the corresponding index values
        # this example has 3 time point, 3 agents, 2 reps and 2 groups (agents 2 and 1 change from group 1 and 0 in the 2 reps)
        rep0 = [[0.000,0.010,0.021],[0.100,0.110,0.121],[0.200,0.210,0.221]]
        rep1 = [[1.000,1.011,1.020],[1.100,1.111,1.120],[1.200,1.211,1.220]]
        timeseries = np.array([rep0,rep1])
        # agent groups
        df_properties = pd.DataFrame({'Agent':[0,1,2,3,4,5],'Rep':[0,0,0,1,1,1],'group':[1,0,1,0,0,1],'Trial':['A','A','A','A','A','A']})
        # get timeseries as dataframe including column for group
        df = time.get_df_timeseries([timeseries],trials=['A'],df_properties=df_properties,variable_name=variable_name,group_names=group_names,downsample_step=1)
        # plot df
        #plot.plot_timeseries(df,variable_name=variable_name,hue='group',style='Trial',palette='colorblind')
    if ge:
        variable_name = 'opinion'
        # define dataframe df[cols=time,agent,opinion,rep,group] with opinion values containing (time,agent,rep,group) using 4 s.f. to represent the corresponding index values
        # this example has 3 time point, 3 agents, 2 reps and 2 groups (agents 2 and 1 change from group 1 and 0 in the 2 reps)
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
    if gs:
        # obtain confidence interval for mean and median
        # for mean, see: https://www.statology.org/confidence-intervals-python/
        # for median, see (but note their example assumes integer values): https://www.statology.org/confidence-interval-for-median/
        valss = [[12, 12, 13, 13, 15, 16, 17, 22, 23, 25, 26, 27, 28, 28, 29], # making these arrays to test effect of downsampling more easily
                 [8, 11, 12, 13, 15, 17, 19, 20, 21, 21, 22, 23, 25, 26, 28]]
        test = 'ANOVA'
        stat,prob = analysis.compare_distributions(valss,test=test) # note that if you compare this with excel, be sure install Analysis ToolPak and use Data Analysis Tools>Anova: Single Factor; I can't make sense of what F.TEST() gives, but it doesn't match with Excel's own Data Analysis result, nor scipy stats, nor online calculators, which all agree with each other from my testing
        print(stat,prob)
        doplot = False # this is unconnected to above statistical test, though the test method is applied to new dummy data
        if doplot: # plot will be saved as png in same folder as script
            df = pd.DataFrame()
            df['x']=[1,2,3,4,5]*4
            df['y']=[234,345,345,45,324,204,352,325,47,340]+[134,245,245,4,224,104,252,225,4,240]
            df['p']=[23,45,56,23,64]+[20,46,52,26,68]+[None]*10
            df['hue']=([0]*5+[1]*5)*2
            df['ci']=[(212,246),(320,376),(311,377),(4,79),(301,355),(200,250),(320,378),(313,338),(40,79),(329,352)]+[(112,146),(220,276),(211,277),(4,79),(201,255),(100,150),(220,278),(213,238),(4,79),(229,252)]
            df['style']=['Mean']*10 + ['Median']*10
            legend = 'simple'#'auto1'
            plot.plot_sensitivity(df, x='x', y='y', hue='hue', linestyle='style', markerstyle='hue', ym='p', ci='ci', base_num = 3, legend = legend, spl='', folder='')