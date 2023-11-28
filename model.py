# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 09:24:31 2023

@author: mark.pogson@liverpool.ac.uk

Modification of Baumann et al. (2020) model to include distinct tendencies as:
    - constant or time-dependent
    - individual or group level
    
Note that only tendency groups were used in the present study, but the code includes scope for:
    - activity
    - controversy
    - homophily
    - sociality
    - reciprocity
    - n_receive
    
See also:
    - functions.py for the functions used in model.py
    - postproc.py for postprocessing to aggregate and compare results
    - postproc_functions.py for the functions used in postproc.py
    
The code includes some features not included/analysed in the present study, e.g. tension. These should be used with caution as they have not been tested/reviewed fully.
    
Note that code comments use PRL as a shorthand for Baumann et al. (2020)
and PRL SM for its supplementary materials

This script outputs csv files to be analysed and plotted using postproc.py

The csv outputs can also be used for independent analysis and plotting of results
"""

import random
import numpy as np
import functions
from datetime import datetime
from joblib import Parallel,delayed

'''
define which simulations to run (see __main__)
for baseline results:
    lim0,lim1 = 0,2 or None,2 for consensus; 2,4 for radicalisation; 4,6 for polarisation; 6,9 or 6,None for minority group trials
for sensitivity analysis:
    lim0,lim1 = 0,3 or None,3 for consensus; 3,6 for radicalisation; 6,9 for polarisation; 9,14 or 9,None for minority group trials
'''
single_lim = None # enter single trial index to run, or None to specify range instead
if single_lim!=None:
    lim0 = single_lim
    lim1 = lim0+1
else:
    lim0,lim1 = None,None # these must correspond to list indices in __main__
SA = False # True to obtain results for sensitivity analysis (see __main__)
minimal_outputs = True # True to limit number of output files and plots (SA has same effect anyway)
ncore = -1 # -1 to use all cores, else specify number of cores available

# model definition (see __main__ to set parameter values for repeated simulations)
def run_model(rep,pre_root_folder,root_folder,n,nt,sociality_value,homophily_value,controversy_value,constant_tendencies,tendency_ng,variant_tendency_kinds,variant_tendency_parameter_changes):
    print(pre_root_folder,root_folder,'start')
    main_folder = 'results' # relative path
    # random seed
    random_seed = 'randint' # options: None (i.e. unreproducibly random), 'randint' (useful for running multiple reps), or numerical value (directly reproducible)
    if random_seed==None:
        random.seed()
    else:
        if random_seed=='randint':
            random_seed = random.randint(0,99999) # i.e. generate random integer random seed for reprodicibility (where the numerical value can be used directly as the seed)
        random.seed(random_seed)
    
    # define model parameters --------------------------------------------------------------------------
    reciprocity_probability = 0.5 # r in paper - probability of links being reciprocal
    n_receive = 10 # maximum number of agents to be influenced by at each iteration (m in PRL SM); this is made into an agent-specific list below
    atm = 1 # adjacency matrix time multiple, i.e. multiple of time step for updating adjacency matrix
    flip_rad = True # True to flip radicalisation so always positive (n.b. this will automatically be made False if variant tendency groups are present, since these will make radicalisation biased so not suitable to be flipped)
    tendencies_init_op = False # if True, this will override and remove any tendency groups, and set tendencies as initial opinions
    
    # define tendency groups
    variant_tendency_parameters = functions.group.get_function_parameters(variant_tendency_parameter_changes)
    overwrite_ids = [] # specify all the agent ids to include in the variant groups (as a flat list, first variant group first), or use an empty list to overwrite the last agents present 
    tendencies,tendency_kinds,tendency_parameters,agent_tendency_parameters,tendency_groups,starti_tendency = functions.group.add_variant_groups(constant_tendencies,tendency_ng,variant_tendency_kinds,variant_tendency_parameters,overwrite_ids=overwrite_ids,n=n)
    
    # define activity groups (i.e. activity above message rate)
    constant_activities = [0] # activities are shifts from baseline message_rates
    activity_ng = 0 # number of agents in each variant group for activities
    variant_activity_kinds = ['spike']
    variant_activity_parameters = [dict(start=nt+1,amplitude=0,duration=200)] # note that start and constant will be defined for each agent (rather than at group level) when agent is excited
    overwrite_ids = [] # specify all the agent ids to include in the variant groups (as a flat list, first variant group first), or use an empty list to overwrite the last agents present
    activities,activity_kinds,activity_parameters,agent_activity_parameters,activity_groups,starti_activity = functions.group.add_variant_groups(constant_activities,activity_ng,variant_activity_kinds,variant_activity_parameters,overwrite_ids=overwrite_ids,n=n)
    
    # define controversy groups
    constant_controversies = [controversy_value]
    controversy_ng = 0 # number of agents in each variant group for controversies
    variant_controversy_kinds = ['spike']
    variant_controversy_parameters = [dict(start=nt+1,amplitude=0,duration=200)] # note that start and constant will be defined for each agent (rather than at group level) when agent is excited
    overwrite_ids = [] # specify all the agent ids to include in the variant groups (as a flat list, first variant group first), or use an empty list to overwrite the last agents present 
    controversies,controversy_kinds,controversy_parameters,agent_controversy_parameters,controversy_groups,starti_controversy = functions.group.add_variant_groups(constant_controversies,controversy_ng,variant_controversy_kinds,variant_controversy_parameters,overwrite_ids=overwrite_ids,n=n)
    
    # define homophily groups
    constant_homophilies = [homophily_value]
    homophily_ng = 0 # number of agents in each variant group for homophilies
    variant_homophily_kinds = ['spike']
    variant_homophily_parameters = [dict(start=nt+1,amplitude=0,duration=200)] # note that start and constant will be defined for each agent (rather than at group level) when agent is excited
    overwrite_ids = [] # specify all the agent ids to include in the variant groups (as a flat list, first variant group first), or use an empty list to overwrite the last agents present 
    homophilies,homophily_kinds,homophily_parameters,agent_homophily_parameters,homophily_groups,starti_homophily = functions.group.add_variant_groups(constant_homophilies,homophily_ng,variant_homophily_kinds,variant_homophily_parameters,overwrite_ids=overwrite_ids,n=n)
    
    # define sociality groups
    constant_socialities = [sociality_value]
    sociality_ng = 0 # number of agents in each variant group for socialities
    variant_sociality_kinds = ['spike']
    variant_sociality_parameters = [dict(start=nt+1,amplitude=0,duration=200)] # note that start and constant will be defined for each agent (rather than at group level) when agent is excited
    overwrite_ids = [] # specify all the agent ids to include in the variant groups (as a flat list, first variant group first), or use an empty list to overwrite the last agents present 
    socialities,sociality_kinds,sociality_parameters,agent_sociality_parameters,sociality_groups,starti_sociality = functions.group.add_variant_groups(constant_socialities,sociality_ng,variant_sociality_kinds,variant_sociality_parameters,overwrite_ids=overwrite_ids,n=n)
    
    # define reciprocity groups
    constant_reciprocities = [reciprocity_probability]
    reciprocity_ng = 0 # number of agents in each variant group for reciprocities
    variant_reciprocity_kinds = ['spike']
    variant_reciprocity_parameters = [dict(start=nt+1,amplitude=0,duration=200)] # note that start and constant will be defined for each agent (rather than at group level) when agent is excited
    overwrite_ids = [] # specify all the agent ids to include in the variant groups (as a flat list, first variant group first), or use an empty list to overwrite the last agents present 
    reciprocities,reciprocity_kinds,reciprocity_parameters,agent_reciprocity_parameters,reciprocity_groups,starti_reciprocity = functions.group.add_variant_groups(constant_reciprocities,reciprocity_ng,variant_reciprocity_kinds,variant_reciprocity_parameters,overwrite_ids=overwrite_ids,n=n)
    
    # define n_receive groups
    constant_n_receives = [n_receive]
    n_receive_ng = 0 # number of agents in each variant group for n_receives
    variant_n_receive_kinds = ['spike']
    variant_n_receive_parameters = [dict(start=nt+1,amplitude=0,duration=200)] # note that start and constant will be defined for each agent (rather than at group level) when agent is excited
    overwrite_ids = [] # specify all the agent ids to include in the variant groups (as a flat list, first variant group first), or use an empty list to overwrite the last agents present 
    n_receives,n_receive_kinds,n_receive_parameters,agent_n_receive_parameters,n_receive_groups,starti_n_receive = functions.group.add_variant_groups(constant_n_receives,n_receive_ng,variant_n_receive_kinds,variant_n_receive_parameters,overwrite_ids=overwrite_ids,n=n)
    
    # agent ids and time steps
    agent_ids = list(range(n))
    times = list(range(nt)) # time measured in iterations
    time_step = 0.01 # time step for opinion dynamics (dt in PRL paper)
    kat = False # True to keep all time points in messaged matrix (e.g. for dynamic analysis of network conections), False to keep only running total for memory reasons
    
    # message rate distribution parameters (activity is used as a synonym here, but elsewhere it refers specifically to changes in message rate)
    activity_exponent = 2.1 # gamma in paper; parameter for activity probability distribution
    activity_base = 0.01 # epsilon in paper; " "
    
    # activity-related parameters (activity is added to message rate to get the effective message rate)
    activity_prox = 0.01 # absolute difference in opinion where activity level can be affected
    activity_amp = 0.05 # activity function amplitude (i.e. stimulation over pre-existing message rate)
    
    # agent properties
    opinion_span = [-1,1] # range for initial opinions
    message_rate_span = [activity_base,1]
    opinions = [random.uniform(*opinion_span) for _ in agent_ids] # randomly assign initial opinions from uniform distribution
    message_rates = [functions.distribution.scalefree_dist(g=activity_exponent, mi=activity_base) for _ in agent_ids]
    
    # set tendencies to be original opinions
    if tendencies_init_op:
        tendencies = opinions[:]
        tendency_ng = 0
    
    # assign groups based on existing property values (useful for later analysis)
    opinion_nog = 2
    message_rate_nog = 4
    opinion_groups,opinion_edges = functions.group.get_groups_from_values(opinions,opinion_span,opinion_nog)
    message_rate_groups,message_rate_edges = functions.group.get_groups_from_values(message_rates,message_rate_span,message_rate_nog,log=True)
    tendency_groups = list(tendency_groups); activity_groups = list(activity_groups) # make these lists (not arrays) for use in tension function
    group_descriptions = ['opinion group','tendency group','message rate group','activity group'] # groups to try out in tension calculation
    
    # update variant groups if ng = 0, to make plotting functions work if they're given the variant groups to plot
    if tendency_ng==0:
        variant_tendency_kinds = []; variant_tendency_parameters = []
    if activity_ng==0:
        variant_activity_kinds = []; variant_activity_parameters = []
        
    # override flip_rad if parameters aren't set for radicalisation, or there are variant groups present which make radicalisation direction no longer 50-50 outcome
    if flip_rad and tendency_ng==0 and controversy_value>0 and homophily_value==0:
        flip_rad = True
    else: # everything else shouldn't be flipped, i.e. not radicalising conditions, or variant groups present, or flip_rad not asked for
        flip_rad = False
    
    # homophily parameters (rarely/never used/altered)
    self_messaging = False # PRL paper reports self-messaging, but email from Baumann says otherwise (results seem OK either way)
    min_dif = 0.01 # only matters if self_messaging is True
    
    # define what counts as recent for connection analysis
    recent_time = int(0.9*nt)
    
    # parameters for social balance (only currently used as a metric rather than part of the model; see Minh Pham et al. (2020) http://dx.doi.org/10.1098/rsif.2020.0752)
    enemy_threshold = 1 # -1 to define friend/enemy ties based on their opinion sign, else give a magnitude of opinion difference where they go from being friends to enemies
    tension_balance = (((n-1)**2+n-1)/2)/((n-2)*(n-1)*n/6)#1 # any positive number to control the relative contribution of triadic social balance to dyadic homophily in the social tension Hamiltonian (g in Minh Pham et al.); I've scaled this by the ratio of triangular an tetrahedal numbers for n-1 and n-2 - see functions __main__ for explanation
    
    # create folder for results
    t0 = datetime.now()
    rep_folder = t0.strftime('%Y_%m_%d_%Hh%Mm%S')
    rep_folder = 'rep'+str(rep+1)+'_'+str(rep_folder)
    folder_name = functions.file.make_folder(main_folder,pre_root_folder,root_folder,rep_folder)
    
    # write parameters to file
    params = [random_seed,flip_rad,n,nt,time_step,atm,activity_exponent,activity_base,activity_prox,activity_amp,self_messaging,min_dif,enemy_threshold,tension_balance,opinion_edges,message_rate_edges,set(tendencies),set(activities),set(controversies),set(homophilies),set(socialities),set(reciprocities),set(n_receives),
              set(tendency_kinds),set(activity_kinds),set(controversy_kinds),set(homophily_kinds),set(sociality_kinds),set(reciprocity_kinds),set(n_receive_kinds),
              set([x.values() for x in tendency_parameters]),set([x.values() for x in activity_parameters]),set([x.values() for x in controversy_parameters]),set([x.values() for x in homophily_parameters]),set([x.values() for x in sociality_parameters]),set([x.values() for x in reciprocity_parameters]),set([x.values() for x in n_receive_parameters]),
              tendency_ng,activity_ng,controversy_ng,homophily_ng,sociality_ng,reciprocity_ng,n_receive_ng]
    param_names = ['random_seed','flip_rad','n','nt','time_step','atm','activity_exponent','activity_base','activity_prox','activity_amp','self_messaging','min_dif','enemy_threshold','tension_balance','opinion_edges','message_rate_edges','tendencies','activities','controversies','homophilies','socialities','reciprocities','n_receives',
                   'tendency_kinds','activity_kinds','controversy_kinds','homophily_kinds','sociality_kinds','reciprocity_kinds','n_receive_kinds','tendency_parameters','activity_parameters','controversy_parameters','homophily_parameters','sociality_parameters','reciprocity_parameters','n_receive_parameters','tendency_ng','activity_ng','controversy_ng','homophily_ng','sociality_ng','reciprocity_ng','n_receive_ng']
    functions.file.write_parameters(params,param_names,folder=folder_name)
    full_params = dict(random_seed=random_seed,flip_rad=flip_rad,n=n,nt=nt,time_step=time_step,atm=atm,activity_exponent=activity_exponent,activity_base=activity_base,activity_prox=activity_prox,activity_amp=activity_amp,self_messaging=self_messaging,min_dif=min_dif,
                       enemy_threshold=enemy_threshold,tension_balance=tension_balance,opinion_edges=opinion_edges,message_rate_edges=message_rate_edges,
                       tendencies=tendencies,activities=activities,controversies=controversies,homophilies=homophilies,socialities=socialities,reciprocities=reciprocities,n_receives=n_receives,
                       starti_tendency=starti_tendency,starti_activity=starti_activity,starti_controversy=starti_controversy,starti_homophily=starti_homophily,starti_sociality=starti_sociality,starti_reciprocity=starti_reciprocity,starti_n_receive=starti_n_receive,
                       tendency_kinds=tendency_kinds,tendency_parameters=tendency_parameters,variant_tendency_kinds=variant_tendency_kinds,variant_tendency_parameters=variant_tendency_parameters,
                       activity_kinds=activity_kinds,activity_parameters=activity_parameters,variant_activity_kinds=variant_activity_kinds,variant_activity_parameters=variant_activity_parameters,
                       controversy_kinds=controversy_kinds,controversy_parameters=controversy_parameters,variant_controversy_kinds=variant_controversy_kinds,variant_controversy_parameters=variant_controversy_parameters,
                       homophily_kinds=homophily_kinds,homophily_parameters=homophily_parameters,variant_homophily_kinds=variant_homophily_kinds,variant_homophily_parameters=variant_homophily_parameters,
                       sociality_kinds=sociality_kinds,sociality_parameters=sociality_parameters,variant_sociality_kinds=variant_sociality_kinds,variant_sociality_parameters=variant_sociality_parameters,
                       reciprocity_kinds=reciprocity_kinds,reciprocity_parameters=reciprocity_parameters,variant_reciprocity_kinds=variant_reciprocity_kinds,variant_reciprocity_parameters=variant_reciprocity_parameters,
                       n_receive_kinds=n_receive_kinds,n_receive_parameters=n_receive_parameters,variant_n_receive_kinds=variant_n_receive_kinds,variant_n_receive_parameters=variant_n_receive_parameters,
                       tendency_groups=tendency_groups,activity_groups=activity_groups,controversy_groups=controversy_groups,homophily_groups=homophily_groups,sociality_groups=sociality_groups,reciprocity_groups=reciprocity_groups,n_receive_groups=n_receive_groups,
                       tendency_ng=tendency_ng,activity_ng=activity_ng,controversy_ng=controversy_ng,homophily_ng=homophily_ng,sociality_ng=sociality_ng,reciprocity_ng=reciprocity_ng,n_receive_ng=n_receive_ng)
    functions.file.pickle_parameters(full_params,folder=folder_name)
    # --------------------------------------------------------------------------------------------------
    print('alpha =',controversy_value,', beta =',homophily_value,', n =',n)
    adjacency_matrix = np.zeros([n,n]) # receivers (i) as rows, senders (j) as columns with probability of messaging from j to i based on homophily
    if kat:
        messaged_matrix = np.zeros([nt,n,n]) # effectively a store of the adjacency matrix at each time step (but either 1 or 0, not a probability or trust value, and it includes reciprocal messaging, which the adjacency matrix doesn't)
    else:
        messaged_matrix = np.zeros([n,n]) # this is a running total of message exchanges, instead of the number at each time step
        recent_matrix = np.zeros([n,n]) # this will be the messaged matrix from recent time onwards (no need to make this here if kat, as it'll be done after the simulation)
    store_opinions = []; store_message_rates = []; store_tendencies = []
    tensions_proximity=[] # this will be [[dyad,triad,hamiltonian] for each time step], using tensions based on enemy_threshold
    tensions_opinion_group=[]; tensions_tendency_group=[]; tensions_message_rate_group=[]; tensions_activity_group=[]
    t_a = -atm # time for adjacency matrix
    for t in times:
        if t%(int(nt/10))==0: print('t = %d'%t + '/%d'%nt) # print every 10% of time simulated
        # store important values
        store_opinions.append(opinions[:]) # record current opinions for plotting
        store_message_rates.append(message_rates[:])
        store_tendencies.append(tendencies[:])
        # update all activities based on time increment (triggering events will be handled with the opinion dynamics)
        activities = [functions.time.update_activity(i,t,message_rates,activities,activity_kinds,activity_parameters,agent_activity_parameters,activity_groups,activity_amp,trigger=False) for i in agent_ids]
        message_rates = [np.clip(message_rate+activity,0,1) for message_rate,activity in zip(store_message_rates[0],activities)] # current message rate is message rate at t=0 plus current activity
        t_a+=atm # adjacency matrix time moves in larger increments than for opinion dynamics
        # update adjacency matrix
        if t==t_a:
            for i in random.sample(agent_ids,n):
                homophily_list,homophily_sum = functions.homophily.sum_homophily(i,opinions,agent_ids,homophilies,min_dif)
                for j in random.sample(agent_ids,n):
                    prob_receive = functions.homophily.sum_probability(i,j,homophily_list,homophily_sum,self_messaging)
                    if prob_receive>=random.random(): # i.e. found a 'neighbour' for this iteration, so can receive message
                        adjacency_matrix[i,j] = prob_receive                                              
        # update opinion dynamics and message rates
        for i in random.sample(agent_ids,n):
            # find most suitable agents to receive messages from
            best_js = list(np.argsort(adjacency_matrix[i,:]))[-1:-(n_receives[i]+1):-1]
            # perform self-shifts in opinion
            opinions[i] += functions.dynamics.self_shift(i,t,opinions,tendency_kinds,tendency_parameters,tendency_groups,time_step) # use this both for my model and the PRL model
            # receive messages from agents j
            for j in random.sample(agent_ids,n):
                if j in best_js and adjacency_matrix[i,j]>0 and message_rates[j]>=random.random():
                    opinions[i] += functions.dynamics.social_shift(i,j,opinions,socialities,controversies,time_step)
                    # record message being received
                    if kat:
                        messaged_matrix[t,i,j] += 1 # this is adding to count for current time step (i.e. from 0)
                    else:
                        messaged_matrix[i,j] += 1 # this is adding to total count from all previous time steps
                        if t>=recent_time: recent_matrix[i,j] += 1
                    # check for changes in activity as a result
                    aod = abs(opinions[i]-opinions[j]) # absolute opinion difference
                    if aod<=activity_prox and activity_kinds[activity_groups[i]]!='constant': # i.e. agent opinion is within range to have its activity affected, and has the correct activity kind
                        message_rates,activities,agent_activity_parameters = functions.time.update_activity(i,t,message_rates,activities,activity_kinds,activity_parameters,agent_activity_parameters,activity_groups,activity_amp,trigger=True)
                # send reciprocal message from i to j
                if adjacency_matrix[i,j]*reciprocities[i]>=random.random() and message_rates[i]>=random.random(): # reciprocal i to j prob is j to i prob multiplied by recipocity prob of i
                    opinions[j]+=functions.dynamics.social_shift(j,i,opinions,socialities,controversies,time_step)
                    # record message being received (no need to update adjacency matrix itself, as messaged_matrix is all that matters for later analysis
                    if kat:
                        messaged_matrix[t,j,i]+=1
                    else:
                        messaged_matrix[j,i]+=1
                        if t>=recent_time: recent_matrix[j,i] += 1
                    # check for changes in activity as a result
                    aod = abs(opinions[j]-opinions[i]) # absolute opinion difference
                    if aod<=activity_prox and activity_kinds[activity_groups[j]]!='constant': # i.e. agent opinion is within range to have its activity affected, and has the correct activity kind
                        message_rates,activities,agent_activity_parameters = functions.time.update_activity(j,t,message_rates,activities,activity_kinds,activity_parameters,agent_activity_parameters,activity_groups,activity_amp,trigger=True)
        # obtain tensions at current time step, based on enemy threshold and different groups
        tensions = functions.tension.get_tension(adjacency_matrix,opinions,enemy_threshold,tension_balance,group_lists=[opinion_groups,tendency_groups,message_rate_groups,activity_groups],group_descriptions=group_descriptions)
        tensions_proximity.append(tensions['proximity']) # i.e. tensions based on enemy_threshold, rather than group
        tensions_opinion_group.append(tensions['opinion group'])
        tensions_tendency_group.append(tensions['tendency group'])
        tensions_message_rate_group.append(tensions['message rate group'])
        tensions_activity_group.append(tensions['activity group'])
    
    # --------------------------------------------------------------------------------------------------
    
    # store final results
    opinion_dynamics = np.array(store_opinions) # time (rows) and agent_ids (columns)
    rate_dynamics = np.array(store_message_rates) # " "
    tendency_dynamics = np.array(store_tendencies) # " "
    if flip_rad and opinion_dynamics[-1,:].mean()<0: # i.e. radicalisation is in negative direction, and flip_rad is True
        opinion_dynamics = -opinion_dynamics # force radicalisation to be in the positive direction
        tendency_dynamics = -tendency_dynamics # also flip tendencies for consistency
        tendencies = [-tendency for tendency in tendencies]
        opinion_groups = functions.group.reverse_indices(opinion_groups)
        tendency_groups = functions.group.reverse_indices(tendency_groups)
    if kat:
        total_matrix = np.sum(messaged_matrix, axis=0) # total number of times column j has sent to row i
        recent_matrix = np.sum(messaged_matrix[recent_time:,:,:], axis=0) # " " in recent time
    else: # recent matrix has already been obtained if not kat
        total_matrix = messaged_matrix # do this for consistency with kat
    sent_counts = np.sum(total_matrix, axis=0) # total number of messages i has sent
    received_counts = np.sum(total_matrix, axis=1) # total number of messages i has received
    #roughnesses = functions.analysis.get_fractal_dimension(opinion_dynamics)
    properties = {'Agent': agent_ids,
        'Initial Opinion': opinion_dynamics[0,:],
        'Final Opinion': opinion_dynamics[-1,:],
        'Message Rate': rate_dynamics[0,:], # just record initial message rate, as they're generally fixed anyway, and rate_dynamics has all the details if required
        'Tendency': tendency_dynamics[0,:], # " "
        'Sent Count': sent_counts,
        'Received Count': received_counts,
        'Tendency Group': tendency_groups,
        'Activity Group': activity_groups, # n.b. this is distinct from message_rate_group (it's group for activity above message rate, e.g. group for time variance)
        'Opinion Group': opinion_groups, # this is grouped by initial opinion
        'Message Rate Group': message_rate_groups, # this is grouped by initial message rate
    }
    #matrix = recent_matrix # matrix to use in plotting connections (could alternatively use a slice of sent_matrix, or adajcency_matrix (which could be stored at each time step like sent_matrix)
    opinion_autocorrelation = functions.analysis.get_autocorrelation(opinion_dynamics)
    rate_autocorrelation = functions.analysis.get_autocorrelation(rate_dynamics)
    
    # write final results to files
    if SA or minimal_outputs: # keep output to minimum for SA, or if requested (only opinion_dynamics and properties are used in postproc.py for most possible analyses)
        results = [opinion_dynamics,properties] # only really need properties for SA, but dynamics may also be useful
        descriptions = ['opinion_dynamics','properties']
    else:
        results = [opinion_dynamics,rate_dynamics,opinion_autocorrelation,rate_autocorrelation,tendency_dynamics,total_matrix,recent_matrix,properties,tensions_proximity,tensions_opinion_group,tensions_tendency_group,tensions_message_rate_group,tensions_activity_group]
        descriptions = ['opinion_dynamics','rate_dynamics','opinion_autocorrelation','rate_autocorrelation','tendency_dynamics','total_matrix','recent_matrix','properties','tensions_proximity','tensions_opinion_group','tensions_tendency_group','tensions_message_rate_group','tensions_activity_group']
    for result,description in zip(results,descriptions):
        functions.file.write_results(results=result, description=description, folder=folder_name)
    
    # plot dynamics
    if not (SA or minimal_outputs):
        MAA = -1#np.argmax(rate_dynamics[0,:]) # -1 to avoid highlighting a particular agent, else give its index
        for description in ['Message Rate','Sent Count','Received Count','Tendency','Tendency Group','Activity Group','Initial Opinion','Final Opinion']:#list(properties):
            functions.plot.plot_dynamics(opinion_dynamics, properties, variable_name='Opinion', description=description, cmap='viridis', kinds=variant_tendency_kinds, parameters=variant_tendency_parameters, groups=tendency_groups, starti=starti_tendency, group_description='Tendency', group_cmap='crest', highlight_agent=MAA, spl=root_folder, folder=folder_name)
           
    print(pre_root_folder,root_folder,'total time = ',datetime.now()-t0)
    
if __name__=='__main__':
    debug = False # True to run sequentially with outputs to terminal, else run in parallel    
    
    # deal with potential input error for stop index due to potentially confusing python index notation for slices
    if lim1==-1: lim1 = None # this is because the nature of slicing can be confusing with the end value; -1 excludes the last entry in list when used as a stop index

    # define baseline minority group values, either used directly in results or altered in sensitivity analysis
    base_ng = 0.05 # minority group size as fraction of all agents (will be converted to int number of agents in code)
    base_amp = 2
    base_dur = 500
    base_start = 0
    base_delay = 100
    
    # define which experiments to run (parameter values will be set below)
    # todo: check what parameters were used on other laptop, as may need to redo fig2 with new sensK
    if not SA:
        # ordered to allow split into slow and fast simulations more easily (all consensuses first, then all radicalisations, then all polarisations)
        pre_root_folders = ['fig1','fig2']*3   +  ['fig4']*2 +['fig5']
        root_folders =   ['a']*2+['b']*2+['c']*2 + ['a','b']  + ['a'] # folder to group reps inside; doubles as subplot label
        # trim and make consistent with SA
        pre_root_folders = pre_root_folders[lim0:lim1]
        root_folders = root_folders[lim0:lim1]
        sens = [None]*len(root_folders)
    else: # note that fig4b for SA is a different sensitivity analysis for fig4a, i.e. it looks at group size rather than function parameterss
        pre_root_folders = [] # these will be populated below based on SApre_root_folders and SAroot_folders below
        root_folders = []
        sens = []
        # ordered to allow split into slow and fast simulations more easily (all consensuses first, then all radicalisations, then all polarisations)
        SApre_root_folders = ['fig1']    +   ['fig2']*2            +     ['fig1']       +      ['fig2']*2       +    ['fig1']       +    ['fig2']*2          +         ['fig4']*4 + ['fig5'] # these are the figures to have sensitivity cases applied to; the actual pre-root and root folder lists will be made longer
        SAroot_folders = ['a_sociality']+['a_sociality','a_tendency']+['b_sociality']+['b_sociality','b_tendency'] + ['c_sociality'] + ['c_sociality','c_tendency'] + ['size','amplitude','duration','start'] + ['delay']   # note these will have indexes added as suffixes for each sensitivity case (to be defined below)
        
        # define sensitivity cases (don't include the base case - this will be included automatically in the postprocessing)
        # note there are 4 case listed for each, but only out of neatness/consistency
        # these lists must be copied exactly in postproc.py to plot the sensitivity analysis
        # fig1/2
        sensK = [0.01,1,6,10] # for fig1 and 2, K values to use in sensitivity analysis
        sensT = [0.01,1.5,3,5]+[10] # for fig2, tendency values to use as +/- (note the order matters for indexing - highlighted new value from original runs with +[10])
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
        for i,sensh in enumerate(single_sens):
            for val in sensh:
                amp = val if i==0 else base_amp 
                dur = val if i==1 else base_dur
                start = val if i==2 else base_start
                sensG_single[i].append([dict(start=start,duration=dur,amplitude=amp)])
        sensG_delay = []
        for val in sens_delay:
            sensG_delay.append([dict(start=base_start,duration=base_dur,amplitude=base_amp),dict(start=val,duration=base_dur,amplitude=-base_amp)])
            
        # add sensitivity cases to pre_root_folder and root_folder lists
        for prf,rf in zip(SApre_root_folders[lim0:lim1],SAroot_folders[lim0:lim1]):
            if prf=='fig1': # this is to make corresponding K sensntivity results to be compared with fig2
                sh = sensK
            elif prf=='fig2':
                if 'sociality' in rf:
                    sh = sensK
                elif 'tendency' in rf:
                    sh = sensT
            elif prf=='fig4':
                if rf=='size':
                    sh = sens_ng
                elif rf=='amplitude':
                    sh = sensG_amplitude
                elif rf=='duration':
                    sh = sensG_duration
                elif rf=='start':
                    sh = sensG_start
            elif prf=='fig5':
                sh = sensG_delay
            nh = len(sh)
            root_folders += [rf]*nh # i.e. repeat subfolder for each item in sensitivity analysis (suffixes will be added in the if statements to define parameters for each subplot)
            pre_root_folders += [prf]*nh # correspondingly repeat pre root folders (which won't need a suffix)
            sens+=sh  

    # error check list lengths
    nsim = len(pre_root_folders)
    assert(len(root_folders)==nsim and len(sens)==nsim)        
    
    # set the number of reps, agents and time steps (a simulation here is a trial, and its reps are dealt with after the trial is defined)
    nreps = [16]*nsim # number of repetitions (2 for testing, 16 for results)
    ns = [500]*nsim # number of agents (10 for testing, 500 for results)
    nts = [1000]*nsim # number of time steps (20 for testing, 1000 for results)

    # define parameter values for each simulation
    constant_tendenciess = [[0]]*nsim # set default as [0] as only fig2 needs anything different and will be specified then
    tendency_ngss = [0]*nsim # set all to default value, to be overridden by particular figure/subplot/sensitivity analysis
    variant_tendency_kindss = [[]]*nsim # this is intentionally an empty list for each sim
    variant_tendency_parameter_changess = [[]]*nsim
    sociality_valuess = [3]*nsim
    controversy_valuess = [None]*nsim # set default as None to force error if not defined for specific plots
    homophily_valuess = [None]*nsim
    
    for i,(pre_root_folder,root_folder) in enumerate(zip(pre_root_folders,root_folders)):
        # define main model parameters
        if pre_root_folder=='fig1' or pre_root_folder=='fig2':
            if SA:
                if sens[i]==None: continue # i.e. want to skip this but retain indexing (as achieved through .index(K) below)
                if 'sociality' in root_folder:
                    if pre_root_folder=='fig2': constant_tendenciess[i] = [-0.5,0.5] # need to define this here, as default is [0] for all trials
                    K = sens[i]
                    sociality_valuess[i] = K
                    root_folders[i] = root_folder+str(sensK.index(K)) # adding index identifier to subfolder name (see sensK for corresponding values - could just use K here, but using index for naming consistency)
                elif 'tendency' in root_folder:
                    T = sens[i] # default K doesn't need redefining (c.f. constant tendencies for sensK)
                    constant_tendenciess[i] = [-T,T]
                    root_folders[i] = root_folder+str(sensT.index(T)) # adding index identifier to subfolder name (see sensT for corresponding values - could just use K here, but using index for naming consistency)
            else: # note there's no need to modify root folder or sociality value, unlike for sensitivity case
                if pre_root_folder=='fig1':
                    constant_tendenciess[i] = [0] # list as many constant tendencies as you want, and they'll be equally and randomly assigned across all agents
                else: # i.e. fig2, with constant tendencies
                    constant_tendenciess[i] = [-0.5,0.5]
            tendency_ngss[i] = 0 # number of agents in each variant group for tendency (set as 0 to have no variant groups)
            variant_tendency_kindss[i] = [] # make sure to match with constant tendency values if appending both
            variant_tendency_parameter_changess[i] = []
            if 'a'==root_folder or 'a_' in root_folder: # i.e. base case or sensitivity case for sensitivity variable
                controversy_valuess[i] = 0.05 # alpha in paper; controversialness of topic
                homophily_valuess[i] = 2 # beta in paper; exponent term for both homophily methods
            elif 'b'==root_folder or 'b_' in root_folder:
                controversy_valuess[i] = 3 # alpha in paper; controversialness of topic
                homophily_valuess[i] = 0 # beta in paper; exponent term for both homophily methods
            elif 'c'==root_folder or 'c_' in root_folder:
                controversy_valuess[i] = 3 # alpha in paper; controversialness of topic
                homophily_valuess[i] = 3 # beta in paper; exponent term for both homophily methods
        elif pre_root_folder=='fig4':
            base_G = [dict(start=base_start,duration=base_dur,amplitude=base_amp)] # baseline pulse parameters
            if SA:
                if sens[i]==None: continue # i.e. want to skip this but retain indexing (as achieved through .index(ng) below)
                variant_tendency_kindss[i] = ['pulse']
                if root_folder=='size': # i.e. varying size of minority group
                    ng = sens[i]
                    root_folders[i] = root_folder+str(sens_ng.index(ng)) # adding index identifier to subfolder name (see sens_ng for corresponding values)
                    tendency_ngss[i] = int(ng*ns[i]) # number of agents in each variant group for tendency (set as 0 to have no variant groups)
                    G = base_G
                else: # i.e. varying function parameters for pulse
                    G = sens[i]
                    if root_folder=='amplitude':
                        sensG = sensG_amplitude
                    elif root_folder=='duration':
                        sensG = sensG_duration
                    elif root_folder=='start':
                        sensG = sensG_start
                    root_folders[i] = root_folder+str(sensG.index(G)) # adding index identifier to subfolder name (see sens_amplitude etc for corresponding values)
                    tendency_ngss[i] = int(base_ng*ns[i]) # number of agents in each variant group for tendency (set as 0 to have no variant groups)
                    if tendency_ngss[i]==0:
                        print('warning: making minority group size non-zero')
                        tendency_ngss[i]=1 # used for testing with small numbers, should never occur in actual sensitivity analysis
            else:
                tendency_ngss[i] = int(base_ng*ns[i]) # number of agents in each variant group for tendency
                if root_folder=='a':
                    G = base_G # parameters for pulse
                    variant_tendency_kindss[i] = ['pulse']
                elif root_folder=='b':
                    G = [dict(start=base_start,duration=1000,amplitude=base_amp)] # parameters for ramp
                    variant_tendency_kindss[i] = ['ramp']
            constant_tendenciess[i] = [0] # list as many constant tendencies as you want, and they'll be equally and randomly assigned across all agents
            controversy_valuess[i] = 3 # alpha in paper; controversialness of topic
            homophily_valuess[i] = 3 # beta in paper; exponent term for both homophily methods
            variant_tendency_parameter_changess[i] = G
        elif pre_root_folder=='fig5':
            if SA:
                if sens[i]==None: continue # i.e. want to skip this but retain indexing (as achieved through .index(G) below)
                G = sens[i]
                root_folders[i] = root_folder+str(sensG_delay.index(G)) # adding index identifier to subfolder name (see sensG_double for corresponding values)
            else: # no need to modify root folder unlike for sensitivty case
                G = [dict(start=base_start,duration=base_dur,amplitude=base_amp),dict(start=base_delay,duration=base_dur,amplitude=-base_amp)]
            variant_tendency_kindss[i] = ['pulse','pulse']
            variant_tendency_parameter_changess[i] = G
            constant_tendenciess[i] = [0] # list as many constant tendencies as you want, and they'll be equally and randomly assigned across all agents
            tendency_ngss[i] = int(0.05*ns[i]) # number of agents in each variant group for tendency (set as 0 to have no variant groups)
            controversy_valuess[i] = 3 # alpha in paper; controversialness of topic
            homophily_valuess[i] = 3 # beta in paper; exponent term for both homophily methods

    # collate parameter values for each simulation in a single list, and duplicate for each repetition
    params = []
    js = [] # this will be the flattened list indices for sims and reps
    j = -1
    for i in range(nsim):
        if not SA or sens[i]!=None:
            for rep in range(nreps[i]):
                j+=1; js.append(j)
                params.append([rep,pre_root_folders[i],root_folders[i],ns[i],nts[i],sociality_valuess[i],homophily_valuess[i],controversy_valuess[i],constant_tendenciess[i],tendency_ngss[i],variant_tendency_kindss[i],variant_tendency_parameter_changess[i]])
    
    # run as many simulations/reps in parallel as ncore permits
    t0 = datetime.now()
    print('simulations started', t0)
    indices = js # note that setting a start and end index is made more complicated by potential for multiple sensitivity cases for each trial
    if debug:
        for i in indices:
            print('****',i,'****')
            run_model(*params[i])
    else:
        Parallel(n_jobs=ncore,verbose=1)(delayed(run_model)(*params[i]) for i in indices)
    t1 = datetime.now()
    print('simulations total time = ',t1-t0)