# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 09:24:31 2023

@author: mark.pogson@liverpool.ac.uk

Modification of Baumann et al. (2020) model to include:
    - exponential decay homophily
    - damping factor
    - agent-specific tendencies
    - group-level constant tendencies and tendency functions
The model can be made to match Baumann et al. (2020) by setting:
    - homophily_method = 'sum'
    - damping_root = 0
    - constant_tendencies = [0]
    - tendency_ng = 0
    
Note that message_rates are a sum of activities and initial message_rates
This means that when using activity groups, the first time point doesn't reflect this sum
It's a tiny quirk of the method, and is simplest to leave in place

It's now possible to specify agent ids when addint variant groups (e.g. apply variant to most active agents, or agents with largest initial opinion magnitudes, etc
As well as tendency and activity, it's now possible to define groups for:
    - controversy
    - homophily
    - sociality
    - reciprocity
    - n_receive
In theory these could all change in response to messages received, like for activity, but not implemented this yet
It wouldn't be much work to do this, but I don't think it's worth pursuing for now

This version makes the model a function which can be run in parallel using joblib
Parameter values for the simulations don't need adjusting before running each one
"""

import random
import numpy as np
import functions
from datetime import datetime
from joblib import Parallel,delayed

def run_model(rep,pre_root_folder,root_folder,n,nt,homophily_method,homophily_value,controversy_value,constant_tendencies,tendency_ng,variant_tendency_kinds,variant_tendency_parameter_changes):
    print(pre_root_folder,root_folder,'start')
    main_folder = 'results' # relative path
    # random seed
    random_seed = 'randint' # None (i.e. unreproducibly random), 'randint' (useful for running multiple reps), or numerical value
    if random_seed==None:
        random.seed()
    else:
        if random_seed=='randint':
            random_seed = random.randint(0,99999) # i.e. generate random integer random seed for reprodicibility
        random.seed(random_seed)
    
    # define model parameters --------------------------------------------------------------------------
    damping_root = 0 # set as 0 to remove damping
    sociality_value = 3 # K in paper
    reciprocity_probability = 0.5 # r in paper - probability of links being reciprocal
    if homophily_method=='exp':
        homophily_value*=10 # reflect difference in homophily forms
    n_receive = 10 # maximum number of agents to be influenced by at each iteration (m in PRL SM); this is made into an agent-specific list below
    atm = 1 # adjacency matrix time multiple, i.e. multiple of time step for updating adjacency matrix
    flip_rad = True # True to flip radicalisation so always positive (n.b. this will automatically be made False if variant tendency groups are present, since these will make radicalisation biased so not suitable to be flipped)
    tendencies_init_op = False # if True, this will override and remove any tendency groups, and set tendencies as initial opinions
    
    # define tendency groups
    variant_tendency_parameters = functions.group.get_function_parameters(variant_tendency_parameter_changes)
    overwrite_ids = [] # specify all the agent ids to include in the variant groups (as a flat list, first variant group first), or use an empty list to overwrite the last agents present 
    tendencies,tendency_kinds,tendency_parameters,agent_tendency_parameters,tendency_groups,starti_tendency = functions.group.add_variant_groups(constant_tendencies,tendency_ng,variant_tendency_kinds,variant_tendency_parameters,overwrite_ids=overwrite_ids,n=n)
    
    # define activity groups (i.e. activity above message rate)
    constant_activities = [0]#[-0.5,0.5] # activities are shifts from baseline message_rates
    activity_ng = 0#int(0.05*n) # number of agents in each variant group for activities
    variant_activity_kinds = ['spike']
    variant_activity_parameters = [dict(start=nt+1,amplitude=0,duration=200)] # note that start and constant will be defined for each agent (rather than at group level) when agent is excited
    overwrite_ids = [] # specify all the agent ids to include in the variant groups (as a flat list, first variant group first), or use an empty list to overwrite the last agents present
    activities,activity_kinds,activity_parameters,agent_activity_parameters,activity_groups,starti_activity = functions.group.add_variant_groups(constant_activities,activity_ng,variant_activity_kinds,variant_activity_parameters,overwrite_ids=overwrite_ids,n=n)
    
    # define controversy groups
    constant_controversies = [controversy_value]#[-0.5,0.5]
    controversy_ng = 0#int(0.05*n) # number of agents in each variant group for controversies
    variant_controversy_kinds = ['spike']
    variant_controversy_parameters = [dict(start=nt+1,amplitude=0,duration=200)] # note that start and constant will be defined for each agent (rather than at group level) when agent is excited
    overwrite_ids = [] # specify all the agent ids to include in the variant groups (as a flat list, first variant group first), or use an empty list to overwrite the last agents present 
    controversies,controversy_kinds,controversy_parameters,agent_controversy_parameters,controversy_groups,starti_controversy = functions.group.add_variant_groups(constant_controversies,controversy_ng,variant_controversy_kinds,variant_controversy_parameters,overwrite_ids=overwrite_ids,n=n)
    
    # define homophily groups
    constant_homophilies = [homophily_value]#[-0.5,0.5]
    homophily_ng = 0#int(0.05*n) # number of agents in each variant group for homophilies
    variant_homophily_kinds = ['spike']
    variant_homophily_parameters = [dict(start=nt+1,amplitude=0,duration=200)] # note that start and constant will be defined for each agent (rather than at group level) when agent is excited
    overwrite_ids = [] # specify all the agent ids to include in the variant groups (as a flat list, first variant group first), or use an empty list to overwrite the last agents present 
    homophilies,homophily_kinds,homophily_parameters,agent_homophily_parameters,homophily_groups,starti_homophily = functions.group.add_variant_groups(constant_homophilies,homophily_ng,variant_homophily_kinds,variant_homophily_parameters,overwrite_ids=overwrite_ids,n=n)
    
    # define sociality groups
    constant_socialities = [sociality_value]#[-0.5,0.5]
    sociality_ng = 0#int(0.05*n) # number of agents in each variant group for socialities
    variant_sociality_kinds = ['spike']
    variant_sociality_parameters = [dict(start=nt+1,amplitude=0,duration=200)] # note that start and constant will be defined for each agent (rather than at group level) when agent is excited
    overwrite_ids = [] # specify all the agent ids to include in the variant groups (as a flat list, first variant group first), or use an empty list to overwrite the last agents present 
    socialities,sociality_kinds,sociality_parameters,agent_sociality_parameters,sociality_groups,starti_sociality = functions.group.add_variant_groups(constant_socialities,sociality_ng,variant_sociality_kinds,variant_sociality_parameters,overwrite_ids=overwrite_ids,n=n)
    
    # define reciprocity groups
    constant_reciprocities = [reciprocity_probability]#[-0.5,0.5]
    reciprocity_ng = 0#int(0.05*n) # number of agents in each variant group for reciprocities
    variant_reciprocity_kinds = ['spike']
    variant_reciprocity_parameters = [dict(start=nt+1,amplitude=0,duration=200)] # note that start and constant will be defined for each agent (rather than at group level) when agent is excited
    overwrite_ids = [] # specify all the agent ids to include in the variant groups (as a flat list, first variant group first), or use an empty list to overwrite the last agents present 
    reciprocities,reciprocity_kinds,reciprocity_parameters,agent_reciprocity_parameters,reciprocity_groups,starti_reciprocity = functions.group.add_variant_groups(constant_reciprocities,reciprocity_ng,variant_reciprocity_kinds,variant_reciprocity_parameters,overwrite_ids=overwrite_ids,n=n)
    
    # define n_receive groups
    constant_n_receives = [n_receive]#[-0.5,0.5]
    n_receive_ng = 0#int(0.05*n) # number of agents in each variant group for n_receives
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
    min_dif = 0.01 # only used if homophily_method = 'sum', and only really matters if self_messaging is True
    
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
    params = [random_seed,homophily_method,flip_rad,n,nt,time_step,atm,damping_root,activity_exponent,activity_base,activity_prox,activity_amp,self_messaging,min_dif,enemy_threshold,tension_balance,opinion_edges,message_rate_edges,set(tendencies),set(activities),set(controversies),set(homophilies),set(socialities),set(reciprocities),set(n_receives),
              set(tendency_kinds),set(activity_kinds),set(controversy_kinds),set(homophily_kinds),set(sociality_kinds),set(reciprocity_kinds),set(n_receive_kinds),
              set([x.values() for x in tendency_parameters]),set([x.values() for x in activity_parameters]),set([x.values() for x in controversy_parameters]),set([x.values() for x in homophily_parameters]),set([x.values() for x in sociality_parameters]),set([x.values() for x in reciprocity_parameters]),set([x.values() for x in n_receive_parameters]),
              tendency_ng,activity_ng,controversy_ng,homophily_ng,sociality_ng,reciprocity_ng,n_receive_ng]
    param_names = ['random_seed','homophily_method','flip_rad','n','nt','time_step','atm','damping_root','activity_exponent','activity_base','activity_prox','activity_amp','self_messaging','min_dif','enemy_threshold','tension_balance','opinion_edges','message_rate_edges','tendencies','activities','controversies','homophilies','socialities','reciprocities','n_receives',
                   'tendency_kinds','activity_kinds','controversy_kinds','homophily_kinds','sociality_kinds','reciprocity_kinds','n_receive_kinds','tendency_parameters','activity_parameters','controversy_parameters','homophily_parameters','sociality_parameters','reciprocity_parameters','n_receive_parameters','tendency_ng','activity_ng','controversy_ng','homophily_ng','sociality_ng','reciprocity_ng','n_receive_ng']
    functions.file.write_parameters(params,param_names,folder=folder_name)
    full_params = dict(random_seed=random_seed,homophily_method=homophily_method,flip_rad=flip_rad,n=n,nt=nt,time_step=time_step,atm=atm,damping_root=damping_root,activity_exponent=activity_exponent,activity_base=activity_base,activity_prox=activity_prox,activity_amp=activity_amp,self_messaging=self_messaging,min_dif=min_dif,
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
    print('homophily method =',homophily_method,', alpha =',controversy_value,', beta =',homophily_value,', n =',n)
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
                if homophily_method=='sum': homophily_list,homophily_sum = functions.homophily.sum_homophily(i,opinions,agent_ids,homophilies,min_dif)
                for j in random.sample(agent_ids,n):
                    if homophily_method=='sum':
                        prob_receive = functions.homophily.sum_probability(i,j,homophily_list,homophily_sum,self_messaging)
                    elif homophily_method=='exp':
                        prob_receive = functions.homophily.exp_probability(i,j,homophilies,opinions,self_messaging)
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
                    opinions[i] += functions.dynamics.social_shift(i,j,opinions,socialities,controversies,damping_root,time_step)
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
                    opinions[j]+=functions.dynamics.social_shift(j,i,opinions,socialities,controversies,damping_root,time_step)
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
#        'Roughness': roughnesses # fractal dimension of each agent's opinion time-series
    }
    matrix = recent_matrix # matrix to use in plotting connections (could alternatively use a slice of sent_matrix, or adajcency_matrix (which could be stored at each time step like sent_matrix)
    opinion_autocorrelation = functions.analysis.get_autocorrelation(opinion_dynamics)
    rate_autocorrelation = functions.analysis.get_autocorrelation(rate_dynamics)
    
    # write final results to files
    results = [opinion_dynamics,rate_dynamics,opinion_autocorrelation,rate_autocorrelation,tendency_dynamics,total_matrix,recent_matrix,properties,tensions_proximity,tensions_opinion_group,tensions_tendency_group,tensions_message_rate_group,tensions_activity_group]
    descriptions = ['opinion_dynamics','rate_dynamics','opinion_autocorrelation','rate_autocorrelation','tendency_dynamics','total_matrix','recent_matrix','properties','tensions_proximity','tensions_opinion_group','tensions_tendency_group','tensions_message_rate_group','tensions_activity_group']
    for result,description in zip(results,descriptions):
        functions.file.write_results(results=result, description=description, folder=folder_name)
    
    # plot dynamics
    MAA = -1#np.argmax(rate_dynamics[0,:]) # -1 to avoid highlighting a particular agent, else give its index
    for description in ['Message Rate','Sent Count','Received Count','Tendency','Tendency Group','Activity Group','Initial Opinion','Final Opinion']:#list(properties):
        functions.plot.plot_dynamics(opinion_dynamics, properties, variable_name='Opinion', description=description, cmap='viridis', kinds=variant_tendency_kinds, parameters=variant_tendency_parameters, groups=tendency_groups, starti=starti_tendency, group_description='Tendency', group_cmap='crest', highlight_agent=MAA, spl=root_folder, folder=folder_name)
        #functions.plot.plot_dynamics(rate_dynamics, properties, variable_name='message rate', description=description, cmap='viridis', kinds=variant_activity_kinds, parameters=variant_activity_parameters, groups=activity_groups, starti=starti_activity, group_description='Activity', group_cmap='rocket', highlight_agent=MAA, spl=root_folder, folder=folder_name)
    
    # plot social tension dynamics
    #functions.plot.plot_timeseries(tss=[tensions_proximity,tensions_opinion_group,tensions_tendency_group,tensions_message_rate_group,tensions_activity_group], types=['proximity']+group_descriptions, description='social tension', title='enemy threshold = %.1f'%enemy_threshold, labels=['dyadic sum','%.3f'%tension_balance+' * triadic sum','Hamiltonian'], xlabel='time', folder=folder_name)
    
    '''
    # commented out certain plots to save time
    # plot autocorrelation for dynamics
    for description in ['message rate','sent count','received count','tendency','tendency group','activity group','initial opinion','final opinion']:#list(properties):
        functions.plot.plot_dynamics(opinion_autocorrelation, properties, autocorr=True, variable_name='opinion', description=description, cmap='viridis', kinds=variant_tendency_kinds, parameters=variant_tendency_parameters, groups=tendency_groups, starti=starti_tendency, group_description='tendency', group_cmap='crest', spl=root_folder, folder=folder_name)
        functions.plot.plot_dynamics(rate_autocorrelation, properties, autocorr=True, variable_name='message rate', description=description, cmap='viridis', kinds=variant_activity_kinds, parameters=variant_activity_parameters, groups=activity_groups, starti=starti_activity, group_description='activity', group_cmap='rocket', spl=root_folder, folder=folder_name)
    
    functions.plot_dynamics_polarity(opinion_dynamics, yt=0, folder=folder_name) # there's a bug in this somewhere but it's rare and not that important anyway
    
    # plot distributions
    # add hue to method, based on postproc_functions
    for description in ['received count','final opinion','activity group','opinion group']:#list(properties):
        functions.plot.plot_distribution(properties, description=description, folder=folder_name)
    
    # plot network diagrams
    for description in ['received count', 'final opinion']:
        functions.plot.plot_network(connection_matrix=matrix, node_values=properties, description=description, cmap='viridis', nticks=5, folder=folder_name)
    
    # plot heatmap of connection frequencies
    functions.plot.plot_heatmap(matrix=matrix, description='sent count', xlabel='agent id (sender)', ylabel='agent id (receiver)', cmap='viridis', folder=folder_name)
    
    # plot property jointplot, including for neighour values
    description='final opinion'
    values = properties[description]
    neighbour_values = functions.analysis.get_neighbour_values(connection_matrix=matrix, values=values)
    functions.plot.plot_joint(values1=values, values2=neighbour_values, description1=description, description2='mean neighbour '+description, folder=folder_name)
    description1 = 'final opinion'; description2 = 'sent count'
    functions.plot.plot_joint(values1=properties[description1], values2=properties[description2], description1=description1, description2=description2, folder=folder_name)
    description2 = 'received count'
    functions.plot.plot_joint(values1=properties[description1], values2=properties[description2], description1=description1, description2=description2, folder=folder_name)
    
    # plot total messages sent between agents, and intrinsic message rate of agents
    for description in list(properties):
        functions.plot.plot_bars(properties, description=description, folder=folder_name)
    '''
        
    print(pre_root_folder,root_folder,'total time = ',datetime.now()-t0)
    
if __name__=='__main__':
    # define number of cores available and folders to use for each simulation
    ncore = -1 # -1 to use all cores
    # note that figure numbers in the paper have all moved down 1, but leaving unchanged in code for simplicity
    pre_root_folders = ['fig4']*2+['fig5'] #['fig1']*3+['fig2']*3+['fig4']*2+['fig5']
    root_folders = ['a','b'] + ['a']    #  ['a','b','c']*2   + ['a','b'] + ['a']  # folder to group reps inside; doubles as subplot label
    
    # set the number of reps, agents, time steps, and the homophily method for each simulation
    nsim = len(pre_root_folders)
    assert(len(root_folders)==nsim)
    nreps = [8]*nsim
    ns = [500]*nsim # number of agents
    nts = [1000]*nsim # number of time steps
    homophily_methods = ['sum']*nsim # 'exp', else 'sum' to use PRL homophily method

    # define parameter values for each simulation
    constant_tendenciess = [None]*nsim
    tendency_ngss = [None]*nsim
    variant_tendency_kindss = [None]*nsim
    variant_tendency_parameter_changess = [None]*nsim
    controversy_valuess = [None]*nsim
    homophily_valuess = [None]*nsim
    for i,(pre_root_folder,root_folder) in enumerate(zip(pre_root_folders,root_folders)):
        # define main model parameters
        if pre_root_folder=='fig1':
            constant_tendenciess[i] = [0] # list as many constant tendencies as you want, and they'll be equally and randomly assigned across all agents
            tendency_ngss[i] = 0 # number of agents in each variant group for tendency (set as 0 to have no variant groups)
            variant_tendency_kindss[i] = [] # make sure to match with constant tendency values if appending both
            variant_tendency_parameter_changess[i] = []
            if root_folder=='a':
                controversy_valuess[i] = 0.05 # alpha in paper; controversialness of topic
                homophily_valuess[i] = 2 # beta in paper; exponent term for both homophily methods
            elif root_folder=='b':
                controversy_valuess[i] = 3 # alpha in paper; controversialness of topic
                homophily_valuess[i] = 0 # beta in paper; exponent term for both homophily methods
            elif root_folder=='c':
                controversy_valuess[i] = 3 # alpha in paper; controversialness of topic
                homophily_valuess[i] = 3 # beta in paper; exponent term for both homophily methods
        elif pre_root_folder=='fig2':
            constant_tendenciess[i] = [-0.5,0.5] # list as many constant tendencies as you want, and they'll be equally and randomly assigned across all agents
            tendency_ngss[i] = 0 # number of agents in each variant group for tendency (set as 0 to have no variant groups)
            variant_tendency_kindss[i] = [] # make sure to match with constant tendency values if appending both
            variant_tendency_parameter_changess[i] = []
            if root_folder=='a':
                controversy_valuess[i] = 0.05 # alpha in paper; controversialness of topic
                homophily_valuess[i] = 2 # beta in paper; exponent term for both homophily methods
            elif root_folder=='b':
                controversy_valuess[i] = 3 # alpha in paper; controversialness of topic
                homophily_valuess[i] = 0 # beta in paper; exponent term for both homophily methods
            elif root_folder=='c':
                controversy_valuess[i] = 3 # alpha in paper; controversialness of topic
                homophily_valuess[i] = 3 # beta in paper; exponent term for both homophily methods
        elif pre_root_folder=='fig4':
            constant_tendenciess[i] = [0] # list as many constant tendencies as you want, and they'll be equally and randomly assigned across all agents
            tendency_ngss[i] = int(0.05*ns[i]) # number of agents in each variant group for tendency (set as 0 to have no variant groups)
            controversy_valuess[i] = 3 # alpha in paper; controversialness of topic
            homophily_valuess[i] = 3 # beta in paper; exponent term for both homophily methods
            if root_folder=='a':
                variant_tendency_kindss[i] = ['pulse'] # make sure to match with constant tendency values if appending both
                variant_tendency_parameter_changess[i] = [dict(start=0,duration=500,amplitude=2)]
            elif root_folder=='b':
                variant_tendency_kindss[i] = ['ramp'] # make sure to match with constant tendency values if appending both
                variant_tendency_parameter_changess[i] = [dict(start=0,duration=1000,amplitude=2)]
        elif pre_root_folder=='fig5':
            constant_tendenciess[i] = [0] # list as many constant tendencies as you want, and they'll be equally and randomly assigned across all agents
            tendency_ngss[i] = int(0.05*ns[i]) # number of agents in each variant group for tendency (set as 0 to have no variant groups)
            controversy_valuess[i] = 3 # alpha in paper; controversialness of topic
            homophily_valuess[i] = 3 # beta in paper; exponent term for both homophily methods
            if root_folder=='a':
                variant_tendency_kindss[i] = ['pulse','pulse'] # make sure to match with constant tendency values if appending both
                variant_tendency_parameter_changess[i] = [dict(start=0,duration=500,amplitude=2),dict(start=100,duration=500,amplitude=-2)]

    # collate parameter values for each simulation in a single list, and duplicate for each repetition
    params = []
    js = [] # this will be the flattened list indices for sims and reps
    j = -1
    for i in range(nsim):
        for rep in range(nreps[i]):
            j+=1; js.append(j)
            params.append([rep,pre_root_folders[i],root_folders[i],ns[i],nts[i],homophily_methods[i],homophily_valuess[i],controversy_valuess[i],constant_tendenciess[i],tendency_ngss[i],variant_tendency_kindss[i],variant_tendency_parameter_changess[i]])
    
    # run as many simulations/reps in parallel as ncore permits
    t0 = datetime.now()
    print('simulations started', t0)
    Parallel(n_jobs=ncore,verbose=1)(delayed(run_model)(*params[j]) for j in js)
    t1 = datetime.now()
    print('all simulations total time = ',t1-t0)