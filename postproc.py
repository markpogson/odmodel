# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 11:28:43 2023

@author: mark.pogson@liverpool.ac.uk

notes:
    - this script collates repetitions of the same simulation, which is termed a trial
    - it also collates multiple trials to allow comparisons of them
    - analysis/plotting options include:
        - timeseries grouped by trial/rep/group
        - correlation of values from different trials/reps/groups (with R and p value of linear regression)
        - tension timeseries for different tension types and agent groups
        - agent values relative to neighbours' values
        - comparison of distributions by trial/rep/group (with ANOVA and Kruskal-Wallis p values)
    - groups can be read in from model outputs or defined post-hoc, e.g. based on adjacencies, opinions at a particular time point, etc
    - note that the presence of multiple trials means Lrep (trial-specific repetition) is an important identifier, as it picks out individual simulations
    - note that the term 'cluster' is used to refer to trials/reps/both, to distinguish these from the terminology of agents 'groups'; 'entity' refers to any of these things
    
this script outputs csv files to be analysed outside the code if wanted
"""
import glob
import os
import pandas as pd
import numpy as np
import functions
import postproc_functions as ppf
from datetime import datetime
from joblib import Parallel,delayed

global var1,var2,var3 # useful to debug variables inside function without returning them
def run_postprocessing(pre_root_folders,root_folders,trials,dynamics_properties,dynamics_variable,dynamics_group,dynamics_group_index,correlation_x,correlation_y,correlation_property,comparison_entity,comparison_variable,comparison_group,comparison_group_indexes,comparison_plot_type,comparison_symmetrical,flip_rad,all_trials,plot_dynamics,plot_correlation,plot_tensions,plot_neighbours,plot_comparison,plot_sensitivity):
    global var1,var2,var3
    #print('postprocessing',trials)
    # define directory and postprocessing options -----------------------------------
    # results folder
    main_folder = 'results'
    
    # define colour palettes to use
    dynamics_palette = 'crest'#'colorblind'#'crest'#'viridis' # palette used for groups in dynamics plots (chosen to match model output continuous scales, but discrete palette is probably more suitable if not showing individual simulation results as well)
    discrete_palette = 'colorblind' # palette used for discrete values, e.g. tension agent groups or comparison agent groups
    
    # folders to collate results from (specify as much detail as you want before *)
    rep_folders = 'rep*' # subfolders containing each rep in root_folder (glob will make use of this to pick out all such folders)
    
    # folder to write collated results to
    output_folder = 'postproc_results'
    
    # downsample resolution for timeseries (note that this will effective drop the last time point in plots, e.g. if nt=1000 and downsample_resolution=5, the displaye time will be from 0 to 800 in steps of 200
    downsample_resolution = 1000 # keep every nt/downsample_resolution time points (where nt is the number of time point), i.e. larger number will keep more time points (automatic limit will be applied to avoid trying to upsample, i.e. interpolate)
    
    # get snapshot properties
    snapshot_variable = '' # empty string to skip snapshot
    snapshot_time = 10
    snapshot_variable_span = [-1,1] # make sure this covers full range required
    snapshot_nog = 2
    snapshot_log = False # True if snapshot_variable follows log distribution
        
    # sort out paths ----------------------------------------------------------------
    # incorporate root folder for reading in and also writing (to organise results more clearly)
    root_foldersh = [rf if rf[0]!='_' else rf[1:] for rf in root_folders] # dealing with _ prefix method
    folderss = [os.path.join(main_folder,pre_root_folder,root_folder,rep_folders) for pre_root_folder,root_folder in zip(pre_root_folders,root_foldersh)] # all folders to read in, e.g. 'a\rep*' for each simulation type being compared
    # this is now effectively error checking based on inputs, e.g. if a particlar folder was specified which doesn't exist
    iremove = [] # indices in terms of folderss, pre_root_folders, root_folders and trials which need to be removed due to not having data
    for i,folder in enumerate(folderss): # print missing folders for clarity
        fold = os.path.split(folder)[0] # directory without rep subfolder
        if not os.path.isdir(fold): # i.e. no folder present for trial
            print('no data for '+fold, ', which is supposed to be Trial '+trials[i])
            iremove.append(i)
    folderss = [x for i,x in enumerate(folderss) if i not in iremove]
    pre_root_folders = [x for i,x in enumerate(pre_root_folders) if i not in iremove]
    root_folders = [x for i,x in enumerate(root_folders) if i not in iremove]
    trials = [x for i,x in enumerate(trials) if i not in iremove]
    
    # create output folder if it doesn't already exist, otherwise just add to/replace existing files in output folder
    if not plot_sensitivity: # want to make folder with simple name which can contain multiple results
        istart = 0
    elif plot_sensitivity=='a' or plot_sensitivity=='b' or plot_sensitivity=='c':
        istart = int(len(trials)/2) # i.e. plotting K sensitivity where fig1 results also have sensitivity
    else: # i.e. just have single control trial, as for T and minority group sensitivities
        istart = 1
    senscases = trials[istart:]
    if istart>0 or plot_dynamics and 'Case' in dynamics_properties: # i.e. sensitivity plot or dynamics plot of sensitivity cases
        prefix = senscases[0].split(' = ')[0]+'='
        senscases = [t.split(' = ')[1] for t in senscases if ' = ' in senscases[0] and ' = ' in t]
        senscases[0] = prefix+senscases[0]
    folder_name = functions.file.make_folder(output_folder,'_'.join(set(pre_root_folders))+'_'+'_'.join(senscases)) # e.g. postproc_results\A_K=1_2_3
    var1 = [trials,pre_root_folders,root_folders]
    
    # read in results to collate ----------------------------------------------------
    print('collating results...')
    # file names in each folder to collate
    files = ['opinion_dynamics','rate_dynamics','tendency_dynamics','total_matrix','recent_matrix','properties','tensions_proximity','tensions_opinion_group','tensions_tendency_group','tensions_message_rate_group','tensions_activity_group']
    
    # make empty lists to store all results in (these have the same names as the individual results in the model)
    opinion_dynamics=[];rate_dynamics=[];tendency_dynamics=[];total_matrix=[];recent_matrix=[];properties=[];tensions_proximity=[];tensions_opinion_group=[];tensions_tendency_group=[];tensions_message_rate_group=[];tensions_activity_group=[]
    results = [opinion_dynamics,rate_dynamics,tendency_dynamics,total_matrix,recent_matrix,properties,tensions_proximity,tensions_opinion_group,tensions_tendency_group,tensions_message_rate_group,tensions_activity_group]
        
    # read in all results and put these in the lists
    do_params = True # flag to read parameters only once
    for file,result in zip(files,results):
        for folders in folderss: # folders are each rep and folderss are each trial
            result.append([])
            for i,folder in enumerate(glob.glob(folders)): # folder is a single rep, and only use nrep of these (see above)
                if i!=1: continue # temp code to use only 1 rep for testing
                try: # include catch in case using results folders which lack certain outputs - should be fine so long as plot_... selections are False for these
                    df = pd.read_csv(os.path.join(folder,'results_'+file+'.csv'))
                    result[-1].append(df)
                except: # either result was not output, or something went wrong for the simulation, e.g. it was cancelled before completion, or it's still in progress
                    if file=='opinion_dynamics' and plot_dynamics:
                        print('no data for opinion dynamics in '+folder)
                    if file=='properties':
                        print('no data for properties in '+folder)
                if do_params: # read in parameters (assume the same across reps/trials, but could store separately if needed in future, e.g. if sensitivity analysis or comparisons made use of these values - currently it's only radicalisation dynamics and tensions which use them
                    params = ppf.file.unpickle_parameters(folder)
                    do_params = False  
    
    # process collated results ---------------------------------------
    # convert results to arrays and/or dataframes
    if plot_dynamics or snapshot_variable:
        opinion_dynamics = ppf.collate.to_array(opinion_dynamics) # (trial,rep,time,agent) # even if not plotting dynamics, use this to obtain nt etc
        # could also do the same with rate_dynamics and tendency_dynamics
        nl,nr,nt,n = opinion_dynamics.shape # number of trials, reps, agents and time points
        downsample_step = int(nt/downsample_resolution) # e.g. downsample_step=1 will keep every timepoint (this would be obtained by setting downsample_resolution=nt)
    if plot_neighbours:
        total_matrix = ppf.collate.to_array(total_matrix) # (trial,rep,agent,agent)
        recent_matrix = ppf.collate.to_array(recent_matrix) # (trial,rep,agent,agent)
    if plot_tensions:
        tensionsss = [ppf.collate.to_array(tensions_proximity), # (trial,rep,time,tension type)
                      ppf.collate.to_array(tensions_opinion_group),
                      ppf.collate.to_array(tensions_tendency_group),
                      ppf.collate.to_array(tensions_message_rate_group),
                      ppf.collate.to_array(tensions_activity_group)]
    df_properties = ppf.collate.to_df_s(properties,trials) # df[columns=agent,properties,trial,rep,trial+rep]; note this capitalises headers too
    
    # add groups based on property values specific to cluster (i.e. trial/rep/both), e.g. above or below median for message rate, roughness, etc - haven't really done much with this
    cluster = 'Lrep' # Lrep is trial-specific rep, so is typically the most useful cluster to define groups with
    df_properties = ppf.analysis.get_groups_from_properties(df_properties, cluster=cluster)
    
    if flip_rad and params['tendency_ng']==0 and np.mean(params['controversies'])>0 and np.mean(params['homophilies'])==0: # i.e. radicalisation with no variant tendency groups, and flip_rad is True
        for l,trial in enumerate(trials):
            is_trial = (df_properties['Trial']==trial)
            reps = sorted(df_properties.loc[is_trial,'Rep'].unique())
            for r,rep in enumerate(reps):
                is_Lrep = is_trial & (df_properties['Rep']==rep) # n.b. Lrep is a property in its own right, but trial and rep iterated separately for indexing reasons
                if df_properties.loc[is_Lrep,'Final Opinion'].mean()<0: # i.e. radicalisation is in negative direction
                    opinion_dynamics[l,r,:,:] = -opinion_dynamics[l,r,:,:] # force radicalisation to be in the positive direction
                    df_properties.loc[is_Lrep,'Initial Opinion'] = -df_properties.loc[is_Lrep,'Initial Opinion']
                    df_properties.loc[is_Lrep,'Final Opinion'] = -df_properties.loc[is_Lrep,'Final Opinion']
                    df_properties.loc[is_Lrep,'Opinion Group'] = functions.group.reverse_indices(df_properties.loc[is_Lrep,'Opinion Group'].values)
                    df_properties.loc[is_Lrep,'Tendency Group'] = functions.group.reverse_indices(df_properties.loc[is_Lrep,'Tendency Group'].values)
    
    # pick out values to add to properties, e.g. initial opinion of most active agent in a given Lrep (agent with highest baseline message rate (MAA)), other MAA properties, mean final global opinion for a given trial, etc
    df_sing,df_properties = ppf.analysis.get_cluster_vals(df_properties, cluster=cluster) # df_properties is updated here to include 'initial opinion group MAA'
    description = 'properties'
    if plot_sensitivity:
        description+='_'+plot_sensitivity # add suffix to the plot, as each sensitivity case will be in the same folder
    elif plot_dynamics:
        description+='_'+plot_dynamics # as above for plot_sensitivity
    functions.file.write_results(results=df_properties,description=description,prefix='postproc',folder=folder_name) # output to csv file to check results

    if plot_dynamics:
        print('plotting dynamics...')
        # make dataframe of timeseries data for each group, as df[columns=time,agent,opinion,rep,group]
        if dynamics_variable=='Opinion':
            timeseriess = opinion_dynamics # n.b. ss is a convenient way of showing this includes trials as well as reps, as used in the get_df_timeseries function
        else:
            print('using opinion anyway...')
            timeseriess = opinion_dynamics
        if dynamics_group!=None and dynamics_group_index!=None: # keep only selected group to plot   
            is_group = (df_properties[dynamics_group]==dynamics_group_index)
            df_properties = df_properties.loc[is_group,:]
        dynamics_propertiesh = [val if val!='Case' else 'Trial' for val in dynamics_properties] # faffing to comply with get_df_properties function (see below)
        df_timeseries = ppf.time.get_df_timeseries(timeseriess,trials,df_properties,dynamics_variable,dynamics_propertiesh,downsample_step=downsample_step)
        sens = False
        if 'Case' in dynamics_properties: # Case is like Trial, but for sensitivity cases
            sens = True
            case_name = 'Case'
            df_timeseries = df_timeseries.rename(columns={'Trial':'Case'}) # need to do this after making timeseries, as get_df_timeseries has Trial hardwired in it
        df_timeseries = df_timeseries.dropna(subset=dynamics_properties)
        for p in dynamics_properties: # force group values to be integer
            if p!='Case' and type(df_timeseries.loc[0,p])!=str: df_timeseries[p] = df_timeseries[p].astype(int)
        if dynamics_group!=None and dynamics_group_index!=None:
            case_name = 'Case ('+dynamics_group+' '+str(dynamics_group_index)+')' # renamed for legend title
        if sens: # rename Trial as Case for legend title (or Case (group index) if being used)
            df_timeseries = df_timeseries.rename(columns={'Case':case_name})
            dynamics_properties = [p if p!='Case' else case_name for p in dynamics_properties] # correspondingly update dynamics_properties
        functions.file.write_results(results=df_timeseries,description='opinion_dynamics'+'_'+plot_dynamics,prefix='postproc',folder=folder_name) # output to csv file to check results
        hue_order = None
        if sens: # sort hue order for plotting
            df_timeseries[['desc','sens']] = df_timeseries[case_name].str.split(' = ',n=1,expand=True)
            df_timeseries['sens'] = df_timeseries['sens'].astype(float)
            df_timeseries = df_timeseries.sort_values(by='sens')#.drop_duplicates(subset='sens') # picking out unique sensitivity values in ascending order
            #hue_order = df_timeseries[case_name].tolist() # these are the hue strings in ascending order of their numerical values
            var2 = df_timeseries
            #var3 = hue_order
        # plot grouped dynamics
        for i,dynamics_property in enumerate(dynamics_properties):
            spl = plot_dynamics
            if i==0: # first property in the list to be plotted using hue alone
                hue = dynamics_property
                style = None
                facet = None
                palette = dynamics_palette # matches the colours used in model outputs from a single simulation
                mlo = False if len(df_timeseries[hue].unique())<4 else True
            elif i==1: # second property in the list, to be combined with the first property by using style
                hue = dynamics_properties[0] # instead of using the 'tendency-opinion' column, use them separately as hue and style
                style = dynamics_property
                facet = None
                palette = discrete_palette # easier to distinguish multiple values
                if pre_root_folders[0]=='fig4': # this is to create fig7 (easiest to do this inside the for loop, as only needed when hue and style are combined for plotting fig4 and 5; this is specifically for fig7 - don't want to change the label otherwise)
                    if not sens: spl = ppf.file.increment_letter(plot_dynamics,-2)
                elif pre_root_folders[0]=='fig5': # again, used for fig7
                    if not sens: spl = ppf.file.increment_letter(plot_dynamics,1)
                mlo = True # move legend outside plot
            else: # 3rd (or later) property in list so use facets for first property and hue and style for 2nd and current property
                facet = dynamics_properties[0]
                hue = dynamics_properties[1] # instead of using the 'tendency-opinion' column, use them separately as hue and style
                style = dynamics_property
                palette = dynamics_palette
            ppf.plot.plot_timeseries(df_timeseries,variable_name=dynamics_variable,title='',hue=hue,hue_order=hue_order,style=style,palette=palette,facet=facet,mlo=mlo,spl=spl,folder=folder_name)
    
    if snapshot_variable: # could do this before plot_dynamics but after get_df_timeseries if wanted to use in dynamics plot groups
        print('getting snapshot of '+snapshot_variable+' at t = %d'%snapshot_time)
        # get properties from a snapshot in dynamics
        t=snapshot_time # this is in original time units (note that timepoints may have been dropped from the downsampled dataframe, e.g. you can't get the last time point unless downsample_resolution=nt)
        downsampled_t = ppf.time.downsample(t,downsample_step)
        snapshot_values = ppf.time.get_snapshot(downsampled_t,df_timeseries,dynamics_variable)
        df_properties['t = %d'%downsampled_t+' '+snapshot_variable] = snapshot_values
        # create groups from the snapshot (note that this uses a function from functions.py, not postproc_functions.py)s
        df_properties['t = %d'%downsampled_t+' '+snapshot_variable+' group'],opinion_edges = functions.group.get_groups_from_values(snapshot_values,snapshot_variable_span,snapshot_nog,log=snapshot_log)
    
    if plot_correlation:
        print('plotting correlation...')
        # plot correlation
        ppf.plot.plot_correlation(df_sing,x=correlation_x,y=correlation_y,hue=correlation_property,spl=plot_correlation,folder=folder_name)
    
    if plot_tensions:
        print('plotting tensions...')
        # get tension timeseries
        df_tensions = ppf.tension.get_df_tensions(tensionsss,trials=trials,columns=['Dyadic','Triadic','Hamiltonian'],group_types=['Proximity','Opinion Group','Tendency Group','Message Rate Group','Activity Group'],downsample_step=downsample_step)
        # plot tensions  - you generally want to filter for specific group types, but to include all, just use filt = df_tensions.iloc[:,0].apply(lambda x:True)
        hue = 'Group Type'
        style = 'Tension Form'
        title = 'Enemy Threshold = %.1f'%params['enemy_threshold']+', $g$ = %.3f'%params['tension_balance']
        #filt = (df_tensions['Group Type']=='Proximity') | (df_tensions['Group Type']=='Opinion Group')
        ppf.plot.plot_timeseries(df_tensions,variable_name='Tension',title=title,hue=hue,style=style,palette=discrete_palette,p0=False,mlo=True,spl=plot_tensions,folder=folder_name)
    
    if plot_neighbours: # could try out facets to show multiple trials together
        print('plotting neighbours...')
        # use connection matrix to get neighbour property values
        property_name = 'Final Opinion'
        variable_name1 = 'Final Opinion'
        variable_name2 = 'Mean Neighbour '+variable_name1
        for l,matrix in zip(trials,recent_matrix): # matrix to use in neighbour analysis (specific to trial)
            isl = (df_properties['Trial']==l)
            neighbour_values = ppf.analysis.get_neighbour_values(matrix, df_properties, property_name)
            df_properties.loc[isl,'Mean Neighbour '+property_name] = neighbour_values
            # plot jointplot
            ppf.plot.plot_joint(df_properties.loc[isl,:],variable_name1,variable_name2,hue=None,folder=folder_name)
            
    if plot_comparison: # note this may make values positive in df_properties (if taking magnitudes for comparison), so do this last
        print('plotting comparison...')
        entities,ne,palette,all_palette = ppf.analysis.get_entities(df_properties,comparison_entity,all_trials)
        hue = comparison_entity
        # make all opinions positive for known symmetrical distributions (to make statistical tests more effective, e.g. for polarisation)
        if comparison_symmetrical:
            df_properties[comparison_variable] = abs(df_properties[comparison_variable])
            new_name = '|'+comparison_variable+'|'
            df_properties = df_properties.rename(columns={comparison_variable: new_name})
            comparison_variable = new_name    
        # define filter for specific group or all agents, depending on selected options
        is_included = (df_properties.iloc[:,0].apply(lambda _: True)) # i.e. select all agents, before further possible filtering
        if comparison_entity=='Trial' and comparison_group: # filter so only agents from the selected group are compared between trials
            for trial,comparison_group_index in zip(trials,comparison_group_indexes): # select specific group for specific trial
                is_trial = (df_properties['Trial']==trial)
                if comparison_group_index!=None: # specified group to compare
                    new_string = trial+' '+comparison_group+' '+str(comparison_group_index)
                    is_group = (df_properties[comparison_group]==comparison_group_index)
                    is_included.loc[is_trial&~is_group] = False # exclude agents in current trial but not in specified group
                    df_properties.loc[is_trial&is_group,'Trial'] = new_string # label for plotting
                    if not all([t==trials[0] for t in trials]): # multiple trials present
                        all_trials = all_trials[:] # making local copy of all_trials, otherwise it gets updated outside the function
                        if trial in all_trials:
                            ix = all_trials.index(trial) # update trials labels to use trial colour, but trial-group label
                            all_trials.remove(trial)
                        else:
                            ix = -1
                        all_trials.insert(ix,new_string) # now have trial-group in place of original trial label
        # perform comparison for all entities together
        count = 0 # used to track increment for subplot label
        if df_properties.loc[is_included,hue].isnull().values.any():
            print('missing data for comparison')
        else:
            bins = 'auto' # could define this instead of binrange to control bins more carefully
            binrange = [df_properties[comparison_variable].min(),df_properties[comparison_variable].max()]
            ppf.plot.plot_distribution(df_properties.loc[is_included,:],variable_name=comparison_variable,hue=hue,palette=palette,ptype=comparison_plot_type,bins=bins,binrange=binrange,spl=ppf.file.increment_letter(plot_comparison,count),folder=folder_name)
            # reset filter before performing pairwise comparisons with reobtained entities for all subgroups
            is_included = (df_properties.iloc[:,0].apply(lambda _: True))
            df_properties['Trial'] = df_properties['Trial'].apply(lambda x: x[:-17] if 'Tendency Group' in x else x) # remove groups from trials so can redo consistently
            for trial in trials: # apply group labels to trials with more than 1 group - could have done this above, but it would have made labelling more fiddly
                is_trial = (df_properties['Trial']==trial)
                if len(df_properties.loc[is_trial,comparison_group].unique())>1: # this is a slightly confusing but easy way to incorporte tendency group into the comparison entity column
                    df_properties.loc[is_trial,'Trial'] = df_properties.loc[is_trial,'Trial'].astype(str)+' '+comparison_group+' '+df_properties.loc[is_trial,'Tendency Group'].astype(str)
            entities,ne,palette,all_palette = ppf.analysis.get_entities(df_properties,comparison_entity,all_trials) # need to re-obtain entities based on above incorporation of group
            if ne>2:
                for i in range(ne-1): # this nested loop ensures i,j is only considered once (comparison is same either way round, so neglect j,i)
                    for j in range(i+1,ne):
                        suffix = '_'+str(i)+'_'+str(j)
                        #folder_name_ij = folder_name+suffix
                        #if not os.path.isdir(folder_name_ij): functions.file.make_folder(folder_name_ij)
                        count+=1
                        is_entities = (df_properties[comparison_entity]==entities[i]) | (df_properties[comparison_entity]==entities[j]) # keep only the two entities to compare
                        palette = [all_palette[entities[i]],all_palette[entities[j]]]
                        spl = plot_comparison#ppf.file.increment_letter(plot_comparison,count)
                        ppf.plot.plot_distribution(df_properties.loc[is_included&is_entities,:],variable_name=comparison_variable,hue=hue,palette=palette,ptype=comparison_plot_type,bins=bins,binrange=binrange,spl=spl,suffix=suffix,folder=folder_name)

    if plot_sensitivity: # like plot_comparison, but taking a range of sensitivity cases and comparing them against a control case
        print('plotting sensitivity...')
        # make all opinions positive for known symmetrical distributions (to make statistical tests more effective, e.g. for polarisation)
        if comparison_symmetrical:
            df_properties[comparison_variable] = abs(df_properties[comparison_variable])
            new_name = '|'+comparison_variable+'|'
            df_properties = df_properties.rename(columns={comparison_variable: new_name})
            comparison_variable = new_name
        # deal with base case (i.e. the baseline for the sensitivity trial)
        df_properties = df_properties.rename(columns={'Trial':'Case'}) # 'Case' should be the comparison_entity, but used Trial above for consistency with other labelling
        var2 = df_properties
        cases = df_properties[comparison_entity].unique().tolist() # list of unique case values in same order as in dataframe
        basei = df_properties.index[df_properties[comparison_entity]==cases[istart]].tolist()[0] # i.e. get the sensitivity base case from the first instance of the sensensitivity cases, as determined by istart
        base_val = df_properties.loc[basei,comparison_entity] # base case is always passed in first, so will always exist in row 0 regardless how many agents/reps/etc are present
        sens_var,base_num = base_val.split(' = ') # this is the sensitivity variable name, e.g. K, and its value for the base case, e.g. 3
        base_num = float(base_num) # convert from string to float
        # initialise lists to store iteration results
        nums = [] # numerical variable values, e.g. K = 1,2,3,...
        averages = [] # mean and median for each case
        average_types = [] # type of average for each case (so can use hue and style in plot)
        cis = [] # 95% confidence interval for each case average
        ps = [] # test probabilities
        test_types = [] # describe the type of test for each p
        trial_types = [] # trial letter for the calculated mean and median
        if istart==1: # i.e. have insensitive control trial
            ind = 0 # index of first row of control trial in df_properties
            is_comp = ppf.analysis.make_filter(df_properties,comparison_entity,cases[ind]) # control trial will always come before sensitivity cases in dataframe, with multiple rows for agents/reps
            condition = (comparison_group and comparison_group_indexes[ind]!=None) # include only specified agent group (this is passed to make filter function)
            is_cont_group = ppf.analysis.make_filter(df_properties,comparison_group,comparison_group_indexes[ind],condition=condition) # a separate control distribution group can be given from the sensitivity case group, but in practice normally the same
            cont_distribution = df_properties.loc[is_comp&is_cont_group,comparison_variable].dropna().values
        for i,(case,comparison_group_index) in enumerate(zip(cases[istart:],comparison_group_indexes[istart:])):
            # if have a sensitive control trial, get corresponding control trial data for current sensitivity case
            if istart!=1: # i.e. have a sensitive control trial
                ind = i # note that i effectively iterates through the sensitivity cases, but also applies to the sensitive control trial index
                is_comp = ppf.analysis.make_filter(df_properties,comparison_entity,cases[ind]) # control trial will always come before sensitivity cases in dataframe, with multiple rows for agents/reps
                condition = (comparison_group and comparison_group_indexes[ind]!=None) # include only specified agent group (this is passed to make filter function)
                is_cont_group = ppf.analysis.make_filter(df_properties,comparison_group,comparison_group_indexes[ind],condition=condition) # a separate control distribution group can be given from the sensitivity case group, but in practice normally the same
                cont_distribution = df_properties.loc[is_comp&is_cont_group,comparison_variable].dropna().values
            # get values for case (which will be the base sensitivity case for first iteration, but will be reordered according to numerical sensitivity values)
            is_case = ppf.analysis.make_filter(df_properties,comparison_entity,case)
            condition = (comparison_group and comparison_group_index!=None) # include only specified agent group (note this uses the comparison group index from the iterable, unlike the control group equivalent above; this could really be done once before iteration for control and case combined as a single filter, but leaving as is since it isn't so slow
            is_group = ppf.analysis.make_filter(df_properties,comparison_group,comparison_group_index,condition=condition)
            case_distribution = df_properties.loc[is_case&is_group,comparison_variable].dropna().values
            # obtain sensitivity numerical value
            case_val = df_properties.loc[is_case&is_group,comparison_entity].values[0]
            if ' = ' in case_val: # this will be true for all the sensitivity cases, and only need one of them to get the case_var which is the same for all
                case_var,case_num = case_val.split(' = ')
                if case_num=='None': continue # just skip, indexing will be fine
                case_num = float(case_num) # note that case_num is simply the parameter value, and equally applies to cont if cont is sensitive to the parameter too
            else:
                case_num = case_val
            # append metrics for sensitivity case
            nums+=[case_num]*2
            mean = np.mean(case_distribution)
            median = np.median(case_distribution)
            averages+=[mean,median]
            average_types+=['Mean','Median']
            cis+=[ppf.analysis.get_ci(case_distribution),ppf.analysis.get_ci(case_distribution,kind='median')]
            distributions = [cont_distribution,case_distribution]
            _,ANOVA_prob = ppf.analysis.compare_distributions(distributions,'ANOVA')
            _,KW_prob = ppf.analysis.compare_distributions(distributions,'K-W')
            if np.isnan(ANOVA_prob): ANOVA_prob = 1 # nan sometimes happens when comparing a distribution with itself
            ps+=[ANOVA_prob,KW_prob]
            test_types+=['ANOVA','K-W']
            # append labels
            if plot_sensitivity=='_a' or plot_sensitivity=='_b' or plot_sensitivity=='_c' or plot_sensitivity=='_d': # fig1c-fig4a comparison
                cont_letter = 'C'
                case_letter = 'G'
            elif plot_sensitivity=='_e': # fig1c-fig5a comparison
                cont_letter = 'C'
                case_letter = 'I'
            elif plot_sensitivity=='_f': # fig4a-fig5a comparison
                cont_letter = 'G'
                case_letter = 'I'
            else: # i.e. fig1-2 comparison
                if plot_sensitivity=='a' or plot_sensitivity=='b' or plot_sensitivity=='c':
                    letter = plot_sensitivity
                else: # i.e. subplot label doesn't match control trial, so use dictionary to covert
                    letter = {'d':'a','e':'b','f':'c'}[plot_sensitivity]
                cont_letter = letter.upper()
                if cont_letter=='A':
                    case_letter = 'D'
                elif cont_letter=='B':
                    case_letter = 'E'
                elif cont_letter=='C':
                    case_letter = 'F'
            trial_types+=[case_letter]*2
            # append corresponding metrics for control case, which may be fixed or sensitive
            nums+=[case_num]*2 # this is still fine even if using a fixed control, as want to plot it at all parameter values either way
            mean = np.mean(cont_distribution)
            median = np.median(cont_distribution)
            averages+=[mean,median]
            average_types+=['Mean','Median']
            cis+=[ppf.analysis.get_ci(cont_distribution),ppf.analysis.get_ci(cont_distribution,kind='median')]
            ps+=[np.nan]*2 # no statistical comparison to perform with self
            test_types+=['']*2
            trial_types+=[cont_letter]*2
        # create dataframe of results, noting that the sensitivity case comes before the control case for each
        df = pd.DataFrame({sens_var:nums,'Average '+comparison_variable:averages,'Average':average_types,'$p$':ps,'Test $p$':test_types,'Trial':trial_types,'ci':cis})
        df = df.sort_values(by=[sens_var,'Trial']) # this orders the rows by sensitivity numbers, then trial, and using whatever base num applies, even if it's a fixed control (in this case, the values will be repeated for clarity in plots)
        var3 = df
        # get consistent colour palette for trials
        _,_,palette,_ = ppf.analysis.get_entities(df,'Trial',all_trials)
        spl = plot_sensitivity.replace('_','') # remove '_' prefix from minority group plots - this was a convenient way to distinguish trials as performed above, without passing further function parameters
        if spl=='a':
            legend = 'simple' # 'simple1' just for line hue, 'simple' for hue,style and marker but without seaborn controlling it, 'auto1' just for line hue/style, 'auto2' just for marker style, and 'auto' for line hue/style and marker style
        else: # to avoid overly cluttered figure, avoid repeating the same line and marker style legend in each plot
            legend = 'simple1'
        ppf.plot.plot_sensitivity(df,x=sens_var,y='Average '+comparison_variable,base_num=base_num,palette=palette,legend=legend,spl=spl,folder=folder_name)
    return # end of run_postprocessing function

if __name__=='__main__': # this looks unnecessarily complicated from the outside, but it just automates postprocessing for multiple figures
    # warning: ensure all results are present! you'll get some error messages if any are missing, but not always as obvious as maybe they should be   
    # particularly confusing is if a folder is missing, e.g. base case folder for SA - you'll get a warning in the terminal, but you still get results which are missing the base case
    ncore = -1 # -1 to use all cores
    debug = False # True to run in series, else parallel (n.b. in parallel there is potential for file access problems)

    lim0,lim1 = 12,None # specify indices for pre_root_folders etc, or None to use all from/to start/end

    SA = True # True if plotting results from a sensitivity analysis, i.e. where multiple versions of the same trial exist and need to be compared
    
    # define trials to plot/compare
    if not SA:
        # figures, subplots and trialss are lists of the same length to enable automated postprocessing of multiple results
        analysis_figures =  ['fig3']*3    + ['fig6']*4                                    + ['fig1']*3  + ['fig2']*3 + ['fig4']*2 + ['fig5']  # these are the figure numbers for the postprocessing output - match them for timeseries (fig 3, 5 and 6) or give new name for comparisons (fig 4 and 7)
        analysis_subplots = ['a','b','c']   + ['a','b','c','d']                          + ['a','b','c'] + ['a','b','c'] + ['a','b','a'] # except for fig6, these labels match the root folder names; see paper for how they work for fig6 which is slightly more confusing due to which results it uses, but it's taken care of in the if statements below
        trialss = [['A','D'],['B','E'],['C','F'],['C','G'],['C','H'],['C','I'],['G','I'], ['A'],['B'],['C'], ['D'],['E'],['F'], ['G'],['H'],['I']] # trials corresponding to the analysis figures and subplotsW
        # create dummy lists for consistency with SA
        nsens = [None]*len(analysis_figures)
        # list for all trials ('at') can be any length, it just depends on all the trials being considered (not all of which need be postprocessed, just named here for colour palette consistency)
        at = ['A','B','C',  'D','E','F',  'G','H',  'I'] # list of all possible trials to compare (to ensure histogram colours are consistent)
    else: # SA as with the non-sensitivity analysis results, but with nsens and other lists also used for each figure/subplot (all these are the same length as trialss)
        # first define the sensitivity values used for the results (must correspond with model.py definitions, in the same order as given - n.b. model.py could use None to preserve consistent ordering, so folder numbers correspond)
        # use None to exclude any but preserve indexing of results

        # fig1/2
        sensK = [0.01,1,6,10] # for fig1 and 2, K values to use in sensitivity analysis (these lists must match those in model.py, or at least use None to preserve indexing used in filenames of results)
        sensT = [0.01,1.5,3,5]+[10] # for fig2, tendency values to use as +/- (full range only used for 2b)
        # fig4
        sens_ng = [0.01,0.1,0.25,0.5]+[0.002] # changing size of single pulse minority group (note that use of ng here is fraction, but it will be converted to integer count, which is what will bes shown in params.txt)
        sens_amplitude = [1,4,6,10]+[0.01,0.1] # numerical value of amplitude as single sensitivity variable
        sens_duration = [50,250,750,1000]+[1,10]
        sens_start = [100,250,500,800]
        sens_delay = [10,50,200,500]+[0]
        sensG_amplitude = [] # this will contain the full pulse parameters for amplitude sensitivity
        sensG_duration = []
        sensG_start = []
        sensG_single = [sensG_amplitude,sensG_duration,sensG_start] # store for pulse parameters
        single_sens = [sens_amplitude,sens_duration,sens_start] # used as iterable to create pulse parameters
        # define figures, subplots, etc (fig2 and 6 to reuse common elements, but will produce fig8 and 9 in the manuscript)
        analysis_figures = ['fig3']*3 +    ['fig3']*3 +    ['fig6']*6 +                          ['fig1']*3 + ['fig2']*6  +       ['fig4']*4      + ['fig5'] # for sensitivity analysis, use figure numbers which match the non-SA plots, to make the parameter definitions easier/consistent (note that the figure numbers themselves don't matter at all for the actual outputs, they're just convenient shorthand here)
        analysis_subplots = ['a','b','c'] + ['d','e','f'] + ['_a','_b','_c','_d','_e','_f'] + ['a_','b_','c_']*2 + ['_a','_b','_c'] + ['a','b','c','d'] + ['a'] # subplot labels to use for each sensitivity plot; note fig3 will produce fig8, and fig6 gives fig9, just used these for convenience with non-SA plots which use the same trials and other parameters; '_' prefix is a convenient way to distinguish trials in plotting method
        nsens =             [len(sensK)]*3 + [len(sensT)]*3 + [len(sens_ng),len(sens_amplitude),len(sens_duration),len(sens_start)]+[len(sens_delay)]*2 + [len(sensK)]*6 + [len(sensT)]*3 + [len(sens_ng),len(sens_amplitude),len(sens_duration),len(sens_start)] + [len(sens_delay)]      # corresponding number of sensitivity cases in sensivity analysis
        # define base case values, which much correspond to model.py values
        baseK = 3 # base K value for constant tendencies
        baseT = 0.5 # base T value for constant tendencies
        base_ng = 0.05 # base ng value for single minority group
        base_amplitude = 2 # base amplitude for single minority group
        base_duration = 500 # base duration for single minority group
        base_start = 0 # base start time for single minority group     
        base_delay = 100 # base time delay for two minority groups
    
    at = ['A','B','C',  'D','E','F',  'G','H',  'I'] # list of all possible trials to compare (to ensure histogram colours are consistent)
    nana = len(analysis_figures)
    assert(len(analysis_subplots)==len(nsens)==nana)
    trialss = [None]*nana
    pre_root_folderss = [None]*nana
    root_folderss = [None]*nana
    dynamics_propertiess = [None]*nana
    dynamics_variabless = [None]*nana
    dynamics_groupss = [None]*nana
    dynamics_group_indexess = [None]*nana
    correlation_xss = [None]*nana
    correlation_yss = [None]*nana
    correlation_propertyss = [None]*nana
    comparison_entityss = [None]*nana
    comparison_variabless = [None]*nana
    comparison_groupss = [None]*nana
    comparison_group_indexess = [None]*nana
    comparison_plot_typess = [None]*nana
    comparison_symmetricalss = [None]*nana
    flip_radss = [None]*nana
    all_trialss = [None]*nana
    plot_dynamicss = [None]*nana
    plot_correlationss = [None]*nana
    plot_tensionss = [None]*nana
    plot_neighbourss = [None]*nana
    plot_comparisonss = [None]*nana
    plot_sensitivityss = [None]*nana
    
    for i,(analysis_figure,analysis_subplot,nsen) in enumerate(zip(analysis_figures,analysis_subplots,nsens)):
        if analysis_figure=='fig1' or analysis_figure=='fig2' or analysis_figure=='fig4' or analysis_figure=='fig5': # this is plotting aggregated timeseries
            if SA:
                dynamics_propertiess[i] = ['Case','Tendency Group']
                if analysis_subplot[0]=='_': # underscore prefix used to indicate tendency sensitivity
                    rootname = analysis_subplot[1:] # i.e. root folder in results for current trial
                    rootname_sens = rootname # used for consistency with minority group, where the sensitivity cases don't use the rootfolder as a prefix
                    case_root = 'tendency' # i.e. sensitivity variable name, which is suffixed to rootname along with an index
                    baseh = baseT # baseline value
                    if analysis_figure=='fig2' and rootname=='b':
                        sensh = sensT # have all sensT values for 'b'
                    else:
                        sensh = sensT[:-1] # have 1 less for 'a' and 'c'
                        nsen -= 1
                    desc = '$T$' # description of case to show in plot legend
                elif analysis_subplot[-1]=='_': # underscore suffix used to indicate sociality sensitivity
                    rootname = analysis_subplot[:-1]
                    rootname_sens = rootname
                    case_root = 'sociality'
                    baseh = baseK
                    sensh = sensK
                    desc = '$K$'
                else: # lack of suffix used to indicate minority group sensitivity
                    rootname = 'a' # sensitivity is only performed for pulse, either single for fig4a, or opposing for fig5a
                    rootname_sens = '' # minority group sensitivity cases don't use the root folder prefix
                    #dynamics_groupss[i] = 'Tendency Group'
                    #dynamics_group_indexess[i] = 0
                    if analysis_figure=='fig4':
                        if analysis_subplot=='a': # i.e. sensitivity for minority group size, expressed as fraction of all agents in minority group
                            case_root = 'size'
                            baseh = base_ng
                            sensh = sens_ng
                            desc = 'Minority Group Size (Fraction)'
                        elif analysis_subplot=='b':
                            case_root = 'amplitude'
                            baseh = base_amplitude
                            sensh = sens_amplitude
                            desc = 'Pulse Amplitude'
                        elif analysis_subplot=='c':
                            case_root = 'duration'
                            baseh = base_duration
                            sensh = sens_duration
                            desc = 'Pulse Duration'
                        elif analysis_subplot=='d':
                            case_root = 'start'
                            baseh = base_start
                            sensh = sens_start
                            desc = 'Pulse Onset Time'
                    elif analysis_figure=='fig5':
                        case_root = 'delay'
                        baseh = base_delay
                        sensh = sens_delay
                        desc = 'Opposing Pulse Time Delay'
                trialss[i] = [desc+' = '+str(baseh)] + [desc+' = '+str(val) for val in sensh]
                root_folderss[i] = [rootname] + [rootname_sens+'_'+case_root+str(i) for i in range(len(sensh))]
                pre_root_folderss[i] = [analysis_figure]+[analysis_figure]*nsen
            else:
                if analysis_figure=='fig1':
                    trialss[i] = analysis_subplot.upper()
                elif analysis_figure=='fig2':
                    trialss[i] = ppf.file.increment_letter(analysis_subplot,3).upper()
                elif analysis_figure=='fig4':
                    trialss[i] = ppf.file.increment_letter(analysis_subplot,6).upper()
                elif analysis_figure=='fig5':
                    trialss[i] = ppf.file.increment_letter(analysis_subplot,8).upper()
                pre_root_folderss[i] = [analysis_figure]
                root_folderss[i] = [analysis_subplot]
                dynamics_propertiess[i] = ['Tendency Group','Opinion Group']#['Tendency Group','Opinion Group']#['Trial','Tendency Group']#['initial opinion group MAA']# you can choose as many as you want, currently they'll be plotted in separate plots using hue within each plot, but could use style to combine on same plot
            dynamics_variabless[i] = 'Opinion'
            if analysis_subplot[0]=='_':
                analysis_subplot = analysis_subplot[1:]
            elif analysis_subplot[-1]=='_':
                analysis_subplot = analysis_subplot[:-1]
            plot_dynamicss[i] = analysis_subplot
            if analysis_figure=='fig1':
                flip_radss[i] = True # this will be applied only if controversy, homophily and mean final opinion allow it
            elif analysis_figure=='fig2':
                flip_radss[i] = True # True to make radicalisation all in the positive direction, regardless of whether the model outputs did this
                if not SA: plot_dynamicss[i] = ppf.file.increment_letter(analysis_subplot,3) # i.e. want a to become d, b to become e, c to become f
            else: # i.e. either fig4 or 5, i.e. time-dependent minority group
                flip_radss[i] = False
                if analysis_figure=='fig4':
                    if not SA: plot_dynamicss[i] = ppf.file.increment_letter(analysis_subplot,2)
                else: # i.e. fig5
                    if not SA: plot_dynamicss[i] = ppf.file.increment_letter(analysis_subplot,1)
        elif analysis_figure=='fig3':
            # trials to postprocess
            if SA: # comparison is between e.g. Trial A and Trial D cont+variants
                if analysis_subplot=='a' or analysis_subplot=='b' or analysis_subplot=='c':
                    rootname = analysis_subplot
                    fig1sens = True
                    case_root = 'sociality'
                    sensh = sensK
                else:
                    rootname = {'d':'a','e':'b','f':'c'}[analysis_subplot]
                    fig1sens = False
                    case_root = 'tendency'
                    if rootname=='b':
                        sensh = sensT # have all T values for 'b'
                    else:
                        sensh = sensT[:-1] # have 1 less for 'a' and 'c'
                        nsen -= 1
                pre_root_folderss[i] = ['fig1']
                if fig1sens:
                    pre_root_folderss[i]+=['fig1']*nsen
                pre_root_folderss[i]+=['fig2']+['fig2']*nsen # need sensitivity cases for fig1 and 2 to match
                scases = [rootname+'_'+case_root+str(i) for i in range(len(sensh))] # sensitivity cases indicated by index suffix
                root_folderss[i] = [rootname] # this will be added to with possible sensitivity comparison cases, then the actual sensitivity cases
                if fig1sens:
                    root_folderss[i]+=scases
                root_folderss[i]+=[rootname]+scases # note that the base case is the fig1 letter regardless of K or T sensitivity
                trialss[i] = [rootname.upper()] # if not fig1sens, this is the only fig1 trial needed, i.e. the fixed base case
                if fig1sens:
                    trialss[i]+=[analysis_subplot.upper()+str(i) for i in range(nsen)] ## i.e. use numerical suffix for 0-tendency cases A-C
                    trialss[i]+=['Sociality $K$ = '+str(baseK)] + ['Sociality $K$ = '+str(val) for val in sensh] # then give case values, e.g. K=1, for the fig2 sensitivity cases
                else:
                    trialss[i]+=['Tendency Magnitude = '+str(baseT)] + ['Tendency Magnitude = '+str(val) for val in sensh] # fig2 sensitivity case values
                plot_sensitivityss[i] = analysis_subplot
                comparison_entityss[i] = 'Case'
                comparison_group_indexess[i] = [None]+[None]*nsen+[None]+[None]*nsen
            else:
                trialss[i] = [analysis_subplot.upper(),ppf.file.increment_letter(analysis_subplot,3).upper()]
                pre_root_folderss[i] = ['fig1','fig2']
                root_folderss[i] = [analysis_subplot]*2
                plot_comparisonss[i] = analysis_subplot
                comparison_entityss[i] = 'Trial' # trial/rep/group to obtain statistical comparison between (note this is distinct from comparison_group)
                comparison_group_indexess[i] = [None,None] # only matters if comparison entity = 'Trial' and comparison_group != ''# this should be a list of group index for each trial, and can mix None with specific values
            comparison_variabless[i] = 'Final Opinion'
            comparison_groupss[i] = 'Tendency Group' # only use if comparison entity = 'Trial' or 'Case' for sensitivity; this will filter for the specified group (see comparison_group_indexes), but can make this blank to ignore groups (same as setting comparison_group_index to None)
            comparison_plot_typess[i] = 'hist' #'violin' #'box'
            if analysis_subplot=='b' or analysis_subplot=='e': # radicalisation not symmetrical; if flip_rad used, final opinions will almost all be positive, but some negatives are likely to be present, so mustn't take magnitude# note that subplot 'e' only occurs with SA
                comparison_symmetricalss[i] = False
            else:
                comparison_symmetricalss[i] = True # set as True where symmetry is known, so can compare symmetrical distributions more meaningfully, e.g. extent of polarisation
            flip_radss[i] = True # True to make radicalisation all in the positive direction, regardless of whether the model outputs did this
            all_trialss[i] = at # define all trials to make colours consistent, even for trials not in current postproc
        elif analysis_figure=='fig6':
            if SA:
                # rename trials to describe base and sensitivity cases (note that trialss[i] will have comparison trial inserted at start of list afterwards)
                if analysis_subplot=='_a': # i.e. sensitivity for minority group size, expressed as fraction of all agents in minority group
                    case_root = 'size'
                    trialss[i] = ['Minority Group Size (Fraction) = '+str(base_ng)] + ['Minority Group Size (Fraction) = '+str(val) for val in sens_ng] # i.e. name to use in place of trial label
                elif analysis_subplot=='_b':
                    case_root = 'amplitude'
                    trialss[i] = ['Pulse Amplitude = '+str(base_amplitude)] + ['Pulse Amplitude = '+str(val) for val in sens_amplitude]
                elif analysis_subplot=='_c':
                    case_root = 'duration'
                    trialss[i] = ['Pulse Time Duration = '+str(base_duration)] + ['Pulse Duration = '+str(val) for val in sens_duration]
                elif analysis_subplot=='_d':
                    case_root = 'start'
                    trialss[i] = ['Pulse Onset Time = '+str(base_start)] + ['Pulse Onset Time = '+str(val) for val in sens_start]
                elif analysis_subplot=='_e' or analysis_subplot=='_f':
                    case_root = 'delay'
                    trialss[i] = ['Opposing Pulse Time Delay = '+str(base_delay)] + ['Opposing Pulse Delay = '+str(val) for val in sens_delay]
                # define root and pre-root folders of results to compare
                inds = range(nsen) #  # i.e. the suffixes for the root folders of the sensitivity cases
                cont_fig = 'fig1' # cont details will be updated for analysis_subplot = '_f'
                cont_root = 'c'
                cont_trial = 'C' # i.e. all comparisons will be against trial C, which is fig1c
                if not (analysis_subplot=='_e' or analysis_subplot=='_f'):
                    fig = 'fig4'
                else:
                    fig = 'fig5' # i.e. want to plot sensitivity results contained in fig5 folder
                    if analysis_subplot=='_f': # i.e. want to compare Trial I delay sensitivity against Trial G
                        cont_fig = 'fig4'
                        cont_root = 'a'
                        cont_trial = 'G'
                pre_root_folderss[i] = [cont_fig]+[fig]+[fig]*nsen # note that this assumes the trial corresponding to fig1 was given as the base trial
                base_root = 'a'
                root_folderss[i] = [cont_root]+[base_root]+[case_root+str(i) for i in inds] # i.e. want to compare fig1c with fig5a/b base plus sensitivity cases
                trialss[i].insert(0,cont_trial) # i.e. all comparisons will be against control trial (generally trial C, which is fig1c, but may also be G, which is fig4a)
                plot_sensitivityss[i] = analysis_subplot
                comparison_entityss[i] = 'Case'
                comparison_group_indexess[i] = [0]+[0]+[0]*nsen # only matters if comparison entity = 'Trial' and comparison_group != ''# this should be a list of group index for each trial, and can mix None with specific values
            else: # i.e. non-SA            
                if analysis_subplot=='c':
                    pre_root_folderss[i] = ['fig1','fig5']
                    root_folderss[i] = ['c','a']
                    trialss[i] = ['C','I']
                elif analysis_subplot=='d':
                    pre_root_folderss[i] = ['fig4','fig5']
                    root_folderss[i] = ['a']*2
                    trialss[i] = ['G','I']
                else: # i.e. subplot a or b
                    pre_root_folderss[i] = ['fig1','fig4']
                    root_folderss[i] = ['c',analysis_subplot]
                    trialss[i] = ['C',ppf.file.increment_letter(analysis_subplot,6).upper()]
                plot_comparisonss[i] = analysis_subplot
                comparison_entityss[i] = 'Trial' # trial/rep/group to obtain statistical comparison between (note this is distinct from comparison_group)
                comparison_group_indexess[i] = [None,0] # only matters if comparison entity = 'Trial' and comparison_group != '' (see above description, where different indexes were used for each trial)
                if analysis_subplot=='d': # i.e. comparing Trials G and I
                    comparison_group_indexess[i] = [0]*2
            comparison_variabless[i] = 'Final Opinion'
            comparison_groupss[i] = 'Tendency Group' # only use if comparison entity = 'Trial' or 'Case' for sensitivity; this will filter for the specified group (see comparison_group_indexes), but can make this blank to ignore groups
            comparison_plot_typess[i] = 'hist' #'violin' #'box'
            all_trialss[i] = at # define all trials to make colours consistent, even for trials not in current postproc
            
    # collate option values for each postprocessing in a single list
    options = [[pre_root_folderss[i],root_folderss[i],trialss[i],dynamics_propertiess[i],dynamics_variabless[i],dynamics_groupss[i],dynamics_group_indexess[i],correlation_xss[i],correlation_yss[i],correlation_propertyss[i],comparison_entityss[i],comparison_variabless[i],comparison_groupss[i],comparison_group_indexess[i],comparison_plot_typess[i],comparison_symmetricalss[i],flip_radss[i],all_trialss[i],plot_dynamicss[i],plot_correlationss[i],plot_tensionss[i],plot_neighbourss[i],plot_comparisonss[i],plot_sensitivityss[i]] for i in range(nana)]
        
    # run as much postprocessing in parallel as ncore permits
    t0 = datetime.now()
    print('postprocessing started', t0)
    indices = range(nana)[lim0:lim1] # this allows only certain figures to be plotted; set either/both lim0,lim1 to None to plot all in either direction
    if debug:
        for i in indices:
            print('****',i,'****')
            print(options[i])
            run_postprocessing(*options[i])
    else:
        Parallel(n_jobs=ncore,verbose=1)(delayed(run_postprocessing)(*options[i]) for i in indices)
    t1 = datetime.now()
    print('all postprocessing total time = ',t1-t0)