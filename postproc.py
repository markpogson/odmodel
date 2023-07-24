# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 11:28:43 2023

@author: mark.pogson@liverpool.ac.uk

notes:
    - this script collates Repetitions of the same simulation, termed a trial
    - it also collates multiple trials to allow comparisons
    - analysis/plotting options include:
        - timeseries grouped by trial/Rep/group
        - correlation of values from different trials/Reps/groups (with R and p value of linear regression)
        - tension timeseries for different tension types and agent groups
        - agent values relative to neighbours' values
        - comparison of distributions by trial/Rep/group (with ANOVA and Kruskal-Wallis p values)
    - groups can be read in from model outputs or defined post-hoc, e.g. based on adjacencies, opinions at a particular time point, etc
    - note that the presence of multiple trials means Lrep (trial-specific Repetition) is an important identifier, as it picks out individual simulations
    - also note that 'cluster' is used to refer to trials/Reps/both, to distinguish these from the terminology of agents 'groups'; 'entity' refers to any of these things
"""
import glob
import os
import pandas as pd
import numpy as np
import functions
import postproc_functions
import seaborn as sns
import string
from datetime import datetime
from joblib import Parallel,delayed

global var
def run_postprocessing(pre_root_folders,root_folders,trials,dynamics_properties,dynamics_variable,correlation_x,correlation_y,correlation_property,comparison_entity,comparison_variable,comparison_group,comparison_group_indexes,comparison_plot_type,comparison_symmetrical,flip_rad,all_trials,plot_dynamics,plot_correlation,plot_tensions,plot_neighbours,plot_comparison):
    global var
    print('postprocessing',trials)
    # define directory and postprocessing options -----------------------------------
    # results folder
    main_folder = 'results'
    
    # define colour palettes to use
    dynamics_palette = 'crest'#'colorblind'#'crest'#'viridis' # palette used for groups in dynamics plots (chosen to match model output continuous scales, but discrete palette is probably more suitable if not showing individual simulation results as well)
    discrete_palette = 'colorblind' # palette used for discrete values, e.g. tension agent groups or comparison agent groups
    
    # folders to collate results from (specify as much detail as you want before *)
    Rep_folders = 'Rep*' # subfolders containing each Rep in root_folder
    
    # folder to write collated results to
    output_folder = 'postproc_results'
    
    # downsample resolution for timeseries (note that this will effective drop the last time point in plots, e.g. if nt=1000 and downsample_resolution=5, the displaye time will be from 0 to 800 in steps of 200
    downsample_resolution = 1000#20 # keep every nt/downsample_resolution time points (where nt is the number of time point), i.e. larger number will keep more time points (will automatically limit to avoid trying to upsample, i.e. interpolate)
    
    # get snapshot properties
    snapshot_variable = '' # empty string to skip snapshot
    snapshot_time = 10
    snapshot_variable_span = [-1,1] # make sure this covers full range required
    snapshot_nog = 2
    snapshot_log = False # True if snapshot_variable follows log distribution
        
    # sort out paths ----------------------------------------------------------------
    # incorporate root folder for reading in and also writing (to organise results more clearly)
    folderss = [os.path.join(main_folder,pre_root_folder,root_folder,Rep_folders) for pre_root_folder,root_folder in zip(pre_root_folders,root_folders)] # all folders to read in, e.g. 'a\Rep*' for each simulation type being compared
    folder_name = os.path.join(output_folder,'_'.join(trials)) # folder to write to, e.g. 'postproc_results\a'
    
    # create output folder if it doesn't already exist, otherwise just add to/Replace existing files in output folder
    if not os.path.isdir(folder_name): functions.file.make_folder(folder_name)
    
    # read in results to collate ----------------------------------------------------
    print('collating results...')
    t0 = datetime.now()
    # file names in each folder to collate
    files = ['opinion_dynamics','rate_dynamics','tendency_dynamics','total_matrix','recent_matrix','properties','tensions_proximity','tensions_opinion_group','tensions_tendency_group','tensions_message_rate_group','tensions_activity_group']
    
    # make empty lists to store all results in (these have the same names as the individual results in the model)
    opinion_dynamics=[];rate_dynamics=[];tendency_dynamics=[];total_matrix=[];recent_matrix=[];properties=[];tensions_proximity=[];tensions_opinion_group=[];tensions_tendency_group=[];tensions_message_rate_group=[];tensions_activity_group=[]
    results = [opinion_dynamics,rate_dynamics,tendency_dynamics,total_matrix,recent_matrix,properties,tensions_proximity,tensions_opinion_group,tensions_tendency_group,tensions_message_rate_group,tensions_activity_group]
        
    # read in all results and put these in the lists
    read_params = True # flag to read parameters only once
    for file,result in zip(files,results):
        for folders in folderss: # folders are each Rep and folderss are each trial
            result.append([])
            for folder in glob.glob(folders): # folder is a single Rep, and only use nRep of these (see above)
                try: # include catch in case using old results folders which lack certain outputs - should be fine so long as plot_... selections are False for these
                    df = pd.read_csv(os.path.join(folder,'results_'+file+'.csv'))
                    result[-1].append(df)
                except: # either something went wrong for the simulation, it was cancelled before completion, or it's still in progress
                    if file=='opinion_dynamics' and plot_dynamics:
                        print('no data for opinion dynamics in '+folder)
                    if file=='properties':
                        print('no data for properties in '+folder)
                if read_params: # read in parameters (assume the same across Reps/trials, but could store separately if needed in future
                    params = postproc_functions.file.unpickle_parameters(folder)
                    read_params = False  
    
    # process collated results ---------------------------------------
    # convert results to arrays and/or dataframes
    if plot_dynamics or snapshot_variable:
        opinion_dynamics = postproc_functions.collate.to_array(opinion_dynamics) # (trial,Rep,time,agent) # even if not plotting dynamics, use this to obtain nt etc
        #rate_dynamics = postproc_functions.collate.to_array(rate_dynamics) # (trial,Rep,time,agent) - not currently using these
        #tendency_dynamics = postproc_functions.collate.to_array(tendency_dynamics) # (trial,Rep,time,agent) - " "
        nl,nr,nt,n = opinion_dynamics.shape # number of trials, Reps, agents and time points
        downsample_step = int(nt/downsample_resolution) # e.g. downsample_step=1 will keep every timepoint (this would be obtained by setting downsample_resolution=nt)
    if plot_neighbours:
        total_matrix = postproc_functions.collate.to_array(total_matrix) # (trial,Rep,agent,agent)
        recent_matrix = postproc_functions.collate.to_array(recent_matrix) # (trial,Rep,agent,agent)
    if plot_tensions:
        tensionsss = [postproc_functions.collate.to_array(tensions_proximity), # (trial,Rep,time,tension type)
                      postproc_functions.collate.to_array(tensions_opinion_group),
                      postproc_functions.collate.to_array(tensions_tendency_group),
                      postproc_functions.collate.to_array(tensions_message_rate_group),
                      postproc_functions.collate.to_array(tensions_activity_group)]
    df_properties = postproc_functions.collate.to_df_s(properties,trials) # df[columns=agent,properties,trial,Rep,trial+Rep]
    df_properties = postproc_functions.file.capitalise_headers(df_properties)
    
    # add groups based on property values specific to cluster (i.e. trial/Rep/both), e.g. above or below median for message rate, roughness, etc - haven't really done much with this
    cluster = 'Lrep' # Lrep is trial-specific Rep, so is typically the most useful cluster to define groups with
    var = df_properties
    df_properties = postproc_functions.analysis.get_groups_from_properties(df_properties, cluster=cluster)
    
    if flip_rad:
        for l,trial in enumerate(trials):
            is_trial = (df_properties['Trial']==trial)
            Reps = sorted(df_properties.loc[is_trial,'Rep'].unique())
            for r,Rep in enumerate(Reps):
                is_Lrep = is_trial & (df_properties['Rep']==Rep) # n.b. Lrep is a property in its own right, but trial and Rep iterated separately for indexing reasons
                if df_properties.loc[is_Lrep,'Final Opinion'].mean()<0 and params['tendency_ng']==0 and np.mean(params['controversies'])>0 and np.mean(params['homophilies'])==0: # i.e. radicalisation is in negative direction, there are no variant tendency groups, and flip_rad is True
                    opinion_dynamics[l,r,:,:] = -opinion_dynamics[l,r,:,:] # force radicalisation to be in the positive direction
                    df_properties.loc[is_Lrep,'Initial Opinion'] = -df_properties.loc[is_Lrep,'Initial Opinion']
                    df_properties.loc[is_Lrep,'Final Opinion'] = -df_properties.loc[is_Lrep,'Final Opinion']
                    df_properties.loc[is_Lrep,'Opinion Group'] = functions.group.reverse_indices(df_properties.loc[is_Lrep,'Opinion Group'].values)
                    df_properties.loc[is_Lrep,'Tendency Group'] = functions.group.reverse_indices(df_properties.loc[is_Lrep,'Tendency Group'].values)
    
    # pick out values to add to properties, e.g. initial opinion of most active agent in a given Lrep (agent with highest baseline message rate (MAA)), other MAA properties, mean final global opinion for a given trial, etc
    df_sing,df_properties = postproc_functions.analysis.get_cluster_vals(df_properties, cluster=cluster) # df_properties is updated here to include 'initial opinion group MAA'
    functions.file.write_results(results=df_properties,description='properties',prefix='postproc',folder=folder_name) # output to csv file to check results
    
    if plot_dynamics:
        print('plotting dynamics...')
        # make dataframe of timeseries data for each group, as df[columns=time,agent,opinion,Rep,group]
        if dynamics_variable=='Opinion':
            timeseriess = opinion_dynamics # n.b. ss is a convenient way of showing this includes trials as well as Reps, as used in the get_df_timeseries function
        else:
            print('using opinion anyway...')
            timeseriess = opinion_dynamics
        df_timeseries = postproc_functions.time.get_df_timeseries(timeseriess,trials,df_properties,dynamics_variable,dynamics_properties,downsample_step=downsample_step)
        functions.file.write_results(results=df_timeseries,description='opinion_dynamics',prefix='postproc',folder=folder_name) # output to csv file to check results
        # plot grouped dynamics
        for i,dynamics_property in enumerate(dynamics_properties):
            spl = plot_dynamics
            if i==0: # first property in the list to be plotted using hue alone
                hue = dynamics_property
                style = None
                palette = dynamics_palette # matches the colours used in model outputs from a single simulation
                mlo = False # legend expected to be small enough to fit inside plot
            else: # subsequent property in the list, to be combined with the first property by using style
                hue = dynamics_properties[0] # instead of using the 'tendency-opinion' column, use them separately as hue and style
                style = dynamics_property
                palette = discrete_palette # easier to distinguish multiple values
                if pre_root_folders[0]=='fig4':
                    spl = postproc_functions.file.increment_letter(plot_dynamics,-2)
                elif pre_root_folders[0]=='fig5':
                    spl = postproc_functions.file.increment_letter(plot_dynamics,1)
                mlo = True # move legend outside plot
            postproc_functions.plot.plot_timeseries(df_timeseries,variable_name=dynamics_variable,title='',hue=hue,style=style,palette=palette,mlo=mlo,spl=spl,folder=folder_name)
    
    if snapshot_variable: # could do this before plot_dynamics but after get_df_timeseries if wanted to use in dynamics plot groups
        print('getting snapshot of '+snapshot_variable+' at t = %d'%snapshot_time)
        # get properties from a snapshot in dynamics
        t=snapshot_time # this is in original time units (note that timepoints may have been dropped from the downsampled dataframe, e.g. you can't get the last time point unless downsample_resolution=nt)
        downsampled_t = postproc_functions.time.downsample(t,downsample_step)
        snapshot_values = postproc_functions.time.get_snapshot(downsampled_t,df_timeseries,dynamics_variable)
        df_properties['t = %d'%downsampled_t+' '+snapshot_variable] = snapshot_values
        # create groups from the snapshot
        df_properties['t = %d'%downsampled_t+' '+snapshot_variable+' group'],opinion_edges = functions.group.get_groups_from_values(snapshot_values,snapshot_variable_span,snapshot_nog,log=snapshot_log)
    
    if plot_correlation:
        print('plotting correlation...')
        # plot correlation
        postproc_functions.plot.plot_correlation(df_sing,x=correlation_x,y=correlation_y,hue=correlation_property,spl=plot_correlation,folder=folder_name)
    
    if plot_tensions:
        print('plotting tensions...')
        # get tension timeseries
        df_tensions = postproc_functions.tension.get_df_tensions(tensionsss,trials=trials,columns=['Dyadic','Triadic','Hamiltonian'],group_types=['Proximity','Opinion Group','Tendency Group','Message Rate Group','Activity Group'],downsample_step=downsample_step)
        # plot tensions  - you generally want to filter for specific group types, but to include all, just use filt = df_tensions.iloc[:,0].apply(lambda x:True)
        hue = 'Group Type'
        style = 'Tension Form'
        title = 'Enemy Threshold = %.1f'%params['enemy_threshold']+', $g$ = %.3f'%params['tension_balance']
        filt = (df_tensions['Group Type']=='Proximity') | (df_tensions['Group Type']=='Opinion Group')
        postproc_functions.plot.plot_timeseries(df_tensions,variable_name='Tension',title=title,hue=hue,style=style,palette=discrete_palette,p0=False,mlo=True,spl=plot_tensions,folder=folder_name)
    
    if plot_neighbours: # could try out facets to show multiple trials together
        print('plotting neighbours...')
        # use connection matrix to get neighbour property values
        property_name = 'Final Opinion'
        variable_name1 = 'Final Opinion'
        variable_name2 = 'Mean Neighbour '+variable_name1
        for l,matrix in zip(trials,recent_matrix): # matrix to use in neighbour analysis (specific to trial)
            isl = (df_properties['Trial']==l)
            neighbour_values = postproc_functions.analysis.get_neighbour_values(matrix, df_properties, property_name)
            df_properties.loc[isl,'Mean Neighbour '+property_name] = neighbour_values
            # plot jointplot
            postproc_functions.plot.plot_joint(df_properties.loc[isl,:],variable_name1,variable_name2,hue=None,folder=folder_name)
            
    if plot_comparison: # note this may make values positive in df_properties (if taking magnitudes for comparison), so do this last
        print('plotting comparison...')
        entities,ne,hue,palette,all_palette = postproc_functions.analysis.get_entities(df_properties,comparison_entity,all_trials)
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
                        ix = all_trials.index(trial) # update trials labels to use trial colour, but trial-group label
                        all_trials.remove(trial)
                        all_trials.insert(ix,new_string) # now have trial-group in place of original trial label
        # perform comparison for all entities together
        count = 0 # used to track increment for subplot label
        if df_properties.loc[is_included,hue].isnull().values.any():
            print('missing data for comparison')
        else:
            binrange = [df_properties[comparison_variable].min(),df_properties[comparison_variable].max()]
            postproc_functions.plot.plot_distribution(df_properties.loc[is_included,:],variable_name=comparison_variable,hue=hue,palette=palette,ptype=comparison_plot_type,binrange=binrange,spl=postproc_functions.file.increment_letter(plot_comparison,count),folder=folder_name)
            # reset filter before performing pairwise comparisons with reobtained entities for all subgroups
            is_included = (df_properties.iloc[:,0].apply(lambda _: True))
            df_properties['Trial'] = df_properties['Trial'].apply(lambda x: x[:-17] if 'Tendency Group' in x else x) # remove groups from trials so can redo consistently
            for trial in trials: # apply group labels to trials with more than 1 group - could have done this above, but it would have made labelling more fiddly
                is_trial = (df_properties['Trial']==trial)
                if len(df_properties.loc[is_trial,comparison_group].unique())>1:
                    df_properties.loc[is_trial,'Trial'] = df_properties.loc[is_trial,'Trial'].astype(str)+' tendency group '+df_properties.loc[is_trial,'Tendency Group'].astype(str)
            entities,ne,hue,palette,all_palette = postproc_functions.analysis.get_entities(df_properties,comparison_entity,all_trials)
            if ne>2:
                for i in range(ne-1): # this nested loop ensures i,j is only considered once (comparison is same either way round, so neglect j,i)
                    for j in range(i+1,ne):
                        folder_name_ij = folder_name+'_'+str(i)+'_'+str(j)
                        if not os.path.isdir(folder_name_ij): functions.file.make_folder(folder_name_ij)
                        count+=1
                        is_entities = (df_properties[comparison_entity]==entities[i]) | (df_properties[comparison_entity]==entities[j]) # keep only the two entities to compare
                        palette = [all_palette[entities[i]],all_palette[entities[j]]]
                        postproc_functions.plot.plot_distribution(df_properties.loc[is_included&is_entities,:],variable_name=comparison_variable,hue=hue,palette=palette,ptype=comparison_plot_type,binrange=binrange,spl=postproc_functions.file.increment_letter(plot_comparison,count),folder=folder_name_ij)
    return # end of run_postprocessing function
                    
if __name__=='__main__': # this looks unnecessarily complicated from the outside, but it just automates postprocessing for multiple figures
    ncore = -1 # -1 to use all cores
    debug = False # True to run in series, else parallel
    lim0,lim1 = None,None # plotting indices to start and end with (None to use all for start and end)
    # note that figure numbers in the paper have all moved down 1, but leaving unchanged in code for simplicity
    analysis_figures =  ['fig3']*3    + ['fig6']*4                                     + ['fig2']*3 + ['fig4']*2 + ['fig5']  # this are the figure numbers for the postprocessing output - match timeseries (fig 3, 5 and 6) or unique for comparisons (fig 4 and 7)
    analysis_subplots = ['a','b','c']   + ['a','b','c','d']                          + ['a','b','c'] + ['a','b','a']
    trialss = [['A','D'],['B','E'],['C','F'],['C','G'],['C','H'],['C','I'],['G','I'], ['D'],['E'],['F'], ['G'],['H'],['I']] # trials corresponding to the analysis figures and subplots
    
    at = ['A','B','C',  'D','E','F',  'G','H',  'I'] # list of all possible trials to compare (to ensure histogram colours are consistent)
    
    nana = len(analysis_figures)
    assert(len(analysis_subplots)==nana and len(trialss)==nana)
    pre_root_folderss = [None]*nana
    root_folderss = [None]*nana
    dynamics_propertiess = [None]*nana
    dynamics_variabless = [None]*nana
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
    
    counter = -1 # used to track number of comparison plots done for fig2 (so can do global and each tendency group)
    groups = [None,0,1] # comparison groups to use according to counter for fig2, i.e. global, -0.5 tendency, +0.5 tendency
    for i,(analysis_figure,analysis_subplot) in enumerate(zip(analysis_figures,analysis_subplots)):
        if analysis_figure=='fig1' or analysis_figure=='fig2' or analysis_figure=='fig4' or analysis_figure=='fig5': # this is plotting aggregated timeseries
            pre_root_folderss[i] = [analysis_figure] # will be corrected for fig5 from c to a
            root_folderss[i] = [analysis_subplot]
            dynamics_propertiess[i] = ['Tendency Group','Opinion Group']#['initial opinion group MAA']# you can choose as many as you want, currently they'll be plotted in separate plots using hue within each plot, but could use style to combine on same plot
            dynamics_variabless[i] = 'Opinion'
            if analysis_figure=='fig1':
                flip_radss[i] = True
                plot_dynamicss[i] = analysis_subplot
            elif analysis_figure=='fig2':
                flip_radss[i] = True # True to make radicalisation all in the positive direction, regardless of whether the model outputs did this
                plot_dynamicss[i] = postproc_functions.file.increment_letter(analysis_subplot,3) # i.e. want a to become d, b to become e, c to become f
            else:
                flip_radss[i] = False
                if analysis_figure=='fig4':
                    plot_dynamicss[i] = postproc_functions.file.increment_letter(analysis_subplot,2)
                else: # i.e. fig5
                    plot_dynamicss[i] = postproc_functions.file.increment_letter(analysis_subplot,1)
        elif analysis_figure=='fig3': # this is plotting histogram comparisons of trials B-C with E-F
            counter+=1
            pre_root_folderss[i] = ['fig1','fig2']
            # trials to postprocess
            root_folderss[i] = [analysis_subplot]*2
            comparison_entityss[i] = 'Trial' # trial/Rep/group to obtain statistical comparison between (note this is distinct from comparison_group)
            comparison_variabless[i] = 'Final Opinion'
            comparison_groupss[i] = 'Tendency Group' # only use if comparison entity = 'Trial'; this will filter for the specified group (see comparison_group_indexes), but can make this blank to ignore groups
            comparison_group_indexess[i] = [None,None]#groups[counter%3]] # only matters if comparison entity = 'Trial' and comparison_group != ''# this should be a list of group index for each trial, and can mix None with specific values
            comparison_plot_typess[i] = 'hist' #'violin' #'box'
            if analysis_subplot=='b':
                comparison_symmetricalss[i] = False # set as True where symmetry is known (but be careful with this) to be present (so can compare symmetrical distributions more effectively, e.g. polarised groups)
            else:
                comparison_symmetricalss[i] = True
            flip_radss[i] = True # True to make radicalisation all in the positive direction, regardless of whether the model outputs did this
            all_trialss[i] = at # define all trials to make colours consistent, even for trials not in current postproc
            plot_comparisonss[i] = postproc_functions.file.increment_letter('a',counter) # i.e. will be a,b,c for comparison of B and E global,group0,group1 and d,e,f for corresponding C and F
        elif analysis_figure=='fig6':
            if analysis_subplot=='c':
                pre_root_folderss[i] = ['fig1','fig5']
                root_folderss[i] = ['c','a']
            elif analysis_subplot=='d':
                pre_root_folderss[i] = ['fig4','fig5']
                root_folderss[i] = ['a']*2
            else:
                pre_root_folderss[i] = ['fig1','fig4']
                root_folderss[i] = ['c',analysis_subplot]
            comparison_entityss[i] = 'Trial' # trial/Rep/group to obtain statistical comparison between (note this is distinct from comparison_group)
            comparison_variabless[i] = 'Final Opinion'
            comparison_groupss[i] = 'Tendency Group' # only use if comparison entity = 'Trial'; this will filter for the specified group (see comparison_group_indexes), but can make this blank to ignore groups
            comparison_group_indexess[i] = [None,0] # only matters if comparison entity = 'Trial' and comparison_group != '' (see above description, where different indexes were used for each trial)
            if analysis_subplot=='d':
                comparison_group_indexess[i] = [0]*2
            comparison_plot_typess[i] = 'hist' #'violin' #'box'
            all_trialss[i] = at # define all trials to make colours consistent, even for trials not in current postproc
            plot_comparisonss[i] = analysis_subplot
            
    # collate options values for each postprocessing in a single list
    options = [[pre_root_folderss[i],root_folderss[i],trialss[i],dynamics_propertiess[i],dynamics_variabless[i],correlation_xss[i],correlation_yss[i],correlation_propertyss[i],comparison_entityss[i],comparison_variabless[i],comparison_groupss[i],comparison_group_indexess[i],comparison_plot_typess[i],comparison_symmetricalss[i],flip_radss[i],all_trialss[i],plot_dynamicss[i],plot_correlationss[i],plot_tensionss[i],plot_neighbourss[i],plot_comparisonss[i]] for i in range(nana)]
        
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