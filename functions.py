# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:10:53 2023

@author: mark.pogson@liverpool.ac.uk

Functions used by the model script
"""

import random
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mpc
import copy
from scipy.interpolate import interp1d
import os
#from pyvis.network import Network
#import networkx as nx
#from hfd import hfd
import pandas as pd
import matplotlib
import pickle
matplotlib.rcParams.update({'font.size': 12})
# n.b. random seed is defined in the model script

class file():
    def make_folder(*names):  # names is a folder name with optional subfolder names, which this function makes and returns the path of
        path = os.path.join(*names)
        path = file.underscore(path)
        if not os.path.isdir(path): os.makedirs(path) # n.b. if folder doesn't exist, new results will just be added to/replace existing ones
        return path
    def underscore(s):
        return s.replace(' ','_').replace('|','').replace('$','').replace('_=_','=') # i.e. replace spaces with underscores and remove any | characters, and $ too for ease of reading
    def write_parameters(params, param_names, folder=''): # write parameters to a human-readable text file, summarising large lists as sets
        with open(os.path.join(folder,'params.txt'),'w') as f:
            for param,param_name in zip(params,param_names):
                f.write(param_name+' = '+str(param)+'\n')
        return
    def pickle_parameters(full_params,folder=''): # dump large lists into a pickled dictionary
        with open(os.path.join(folder,'params.pkl'),'wb') as f:
            pickle.dump(full_params,f)
        return
    def write_results(results,description='',prefix='results',folder=''): # write results to csv file
        df = pd.DataFrame(results) # this will automatically create a header row of indices if there isn't already one (e.g. a dictionary will use the keys as headers)
        df.to_csv(os.path.join(folder,prefix+'_'+file.underscore(description)+'.csv'),index=False)
        return
        
class distribution():
    def pdf(x,g=2.1,mi=0.01):
        return (1-g)/(1-mi**(1-g))*x**(-g)
    def scalefree_dist(g=2.1,mi=0.01): # generate random number from dist F(x) ~ x^(-g) in range [mi,1]; use this to give scale-free network
        # see F(a) in PRL supplementary information
        if g==1:
            return('g must not equal 1')
        pdf_max = distribution.pdf(mi,g,mi) # max of pdf in range
        for _ in range(999999): # arbitrary limit to number of attempts tp obtain suitable random number
            x = random.uniform(mi,1) # i.e. maximum possible message rate is 1, minimum is mi
            if random.random() <= distribution.pdf(x,g,mi)/(pdf_max+1): # i.e. use x if random number falls under rescaled pdf curve (rescaling is arbitrary so long as max<1; smaller curve will result in taking more iterations to get result, but distribution unaffected)
                return x

class homophily():
    def sum_homophily_function(opinion1,opinion2,homophily_exponent,min_dif=0.01):    # function for |x_j-x_i|^(-beta)
        if abs(opinion1-opinion2)<=min_dif: # avoid raising 0 or very small number to -ve power
            opinion1+=min_dif
        return abs(opinion1-opinion2)**(-homophily_exponent)
    def sum_homophily(i,opinions,agent_ids,homophilies,min_dif): # obtain all terms required for the sum homophily probability
        homophily_list = [homophily.sum_homophily_function(opinions[i],opinions[j],homophilies[i],min_dif) for j in agent_ids]
        homophily_sum = sum(homophily_list)
        return homophily_list,homophily_sum
    def sum_probability(i,j,homophily_list,homophily_sum,self_messaging=False): # probability of message received by i from j
        if i==j and not self_messaging:
            prob_receive=0 # removing self-messaging has a pretty big effect on results - the paper says it's included, but the author emailed to say it wasn't, but dynamics results look like it was
        else:
            prob_receive = homophily_list[j] / homophily_sum
        return prob_receive
    def exp_probability(i,j,homophilies,opinions,self_messaging=False): # used to test alternative probability method, but note it doesn't scale sensibly, so relies far more on the message limit m
        if i==j and not self_messaging:
            prob_receive=0
        else:
            dif = abs(opinions[i]-opinions[j])
            prob_receive = np.exp(-homophilies[i]*dif)
        return prob_receive

class dynamics():
    def self_shift(i,t,opinions,tendency_kinds,tendency_parameters,groups,time_step): # i moves opinion to towards tendency
        tendency = time.time_dependent_function(t,kind=tendency_kinds[groups[i]],parameters=tendency_parameters[groups[i]])
        return time_step*(tendency-opinions[i])
    def social_shift(i,j,opinions,socialities,controversies,time_step): # obtain single term in social summation, scaled by time step (could do this once all terms have been obtained, but neater this way)
        return time_step*socialities[i]*np.tanh(controversies[i]*opinions[j])
    def social_shift_piecewise(i,j,opinions,socialities,controversies,time_step): # function used to test out other forms of damping, but approach is something of a dead end
        if (opinions[i]>0 and opinions[j]>opinions[i]) or (opinions[i]<0 and opinions[j]<opinions[i]):
            shift = np.tanh(controversies[i]*opinions[j])
        else:
            shift = np.tanh(controversies[i]*(opinions[j]-opinions[i])) # need to test out a curve which is closer to the above but doesn't cause confusing shifts
        return time_step*socialities[i]*shift

class time():
    # note that all time-dependent functions accept the same parameters for convenience
    # jump is distinct from constant in that it changes with triggering events
    def constant_function(t,jump=0,constant=0,start=None,duration=None,amplitude=None,wavelength=None):
        val = constant
        return val    # this looks counterintuitive, but constant is dealt with in the calling function, and this is just for consistency with the other functions
    def oscillation_function(t,jump=0,constant=0,start=0,duration=100,amplitude=1,wavelength=0.005*2*np.pi):
        val = constant
        end = start+duration
        if t>=start and t<=end:
            val += jump+amplitude*np.sin(wavelength*(t-start))
        return val
    def pulse_function(t,jump=0,constant=0,start=0,duration=100,amplitude=1,wavelength=None):
        val = constant
        end = start+duration
        if t>=start and t<=end:
            val += jump+amplitude
        return val
    def ramp_function(t,jump=0,constant=0,start=0,duration=100,amplitude=1,wavelength=None):
        # jump is an addition to the constant (used for activity over baseline)
        # start is the time point that the ramp begins
        # duration is the time length of the ramp
        # amplitude is the height of the ramp at the end of the duration, above jump+constant
        def f(t): return jump+gradient*(t-start)
        val = constant # note this is added to by f(t)
        gradient = amplitude/duration
        end = start+duration
        if t>end:
            val += f(end)
        elif t>=start:
            val += f(t)
        return val
    def spike_function(t,jump=0,constant=0,start=0,duration=100,amplitude=1,wavelength=None):
        val = constant
        decay = 5/duration # approximate decay for function to drop to 5% of its intial value within the given duration
        if t>=start:
            val += (jump+amplitude)*np.exp(-decay*(t-start))
        return val
    def time_dependent_function(t,kind='constant',parameters=dict(jump=0,constant=0)):
        function_forms = dict(constant=time.constant_function,oscillation=time.oscillation_function,
                              pulse=time.pulse_function,ramp=time.ramp_function,spike=time.spike_function)
        value = function_forms[kind](t, **parameters)
        return value
    def update_activity(i,t,message_rates,activities,activity_kinds,activity_parameters,agent_activity_parameters,activity_groups,amp,trigger=False):
        k = activity_groups[i]
        if trigger: # i.e. activity change triggered (e.g. when message received); this could also be made capable of updating group-level parameters if wanted
            agent_activity_parameters[i]['start'] = t
            agent_activity_parameters[i]['jump']=activities[i]
            agent_activity_parameters[i]['amplitude']=amp
        params = {**activity_parameters[k],**agent_activity_parameters[i]}
        activities[i] = time.time_dependent_function(t,activity_kinds[k],params)
        if trigger: # return updated message rate, activity and agent activity parameters
            nmr = message_rates[i]+activities[i]
            message_rates[i] = np.clip(nmr,0,1) # new message rate clipped between 0 and 1
            return message_rates,activities,agent_activity_parameters
        else:
            return activities[i]
   
class group():
    def get_groups_from_values(vals,span,nog,log=False):
        # vals are the agent values, e.g. initial opinion
        # span is the full possible range of values, e.g. [-1,1]
        # nog is the number of groups to define within the values, equally spaced in the span of values
        # log=True uses logarithmic spacing for the value edges to define the groups from
        def get_index(val,edges):
            for i,edge in enumerate(edges[1:]): # [1:] as first edge is minimum parameter value, and want group indexing to start at 0
                if val<=edge: # this gives bin edges as (min,max], except end bins which are [min,max]
                    return i
        if not log:
            edges = np.linspace(*span, nog+1) # +1 to include both outer edges
        else:
            edges = np.logspace(*np.log10(span), nog+1) # logspace uses base 10 by default
        groups = [get_index(val,edges) for val in vals] # group index for each value
        return groups,edges
    def add_groups(ng,variant_values,existing_list,overwrite_ids=[]): # overwrite=True to overwrite values at end of existing list (used to update agent-level lists), else False to append (used for group-level lists)
        # ng is the number of agent in each group (all the same size), variant_values are the agent values to fix for each group, and property_list is the list of agent properties to update
        el = np.array(existing_list[:])
        if ng>0:
            gvs = [[gv for _ in range(ng)] for gv in variant_values]
            if overwrite_ids:
                el[overwrite_ids] = sum(gvs,[])
                el = list(el)
            else:
                el = list(el) + sum(gvs,[])
        return el
    def add_variant_groups(input_values,ng,variant_kinds,variant_parameters,overwrite_ids,n):
        # first create agent values from constant values; this can be as many (up to an including all agents) or as few (up to and including 0) as you want
        ncv = len(input_values)
        groups = [random.randint(0,ncv-1) for _ in range(n)] # this gives index 0 to the first constant value, 1 to the second, etc
        kinds = ['constant' for _ in range(ncv)]
        parameters = [dict(constant=input_values[k]) for k in range(ncv)]
        # sort out number and indexing of variant groups to add
        n_groups = len(variant_kinds)
        starti = ncv if n>n_groups*ng else 0
        variant_groups=list(range(starti,starti+n_groups)) # group indices
        if not overwrite_ids: # create a list of ids at the end of all agent ids (could have used [randint(0,n-1) for _ in len(variant_values)] to overwrite random agents
            overwrite_ids = list(range(n-len(variant_groups)*ng,n))
        # add variant groups to groups list
        groups = group.add_groups(ng, variant_values=variant_groups, existing_list=groups, overwrite_ids = overwrite_ids)
        # add variant kinds and parameters
        if ng>0: # ng is number of agents in each variant group
            kinds = group.add_groups(1, variant_values=variant_kinds, existing_list=kinds, overwrite_ids=[])
            parameters = group.add_groups(1, variant_values=variant_parameters, existing_list=parameters, overwrite_ids=[])
        # use kinds and parameters to create values list
        values = [time.time_dependent_function(t=0,kind=kinds[groups[i]],parameters=parameters[groups[i]]) for i in range(n)]
        # inheret group parameters for each agent; this allows for agent-specific behaviour, e.g. activity spikes
        agent_parameters = [parameters[groups[i]] for i in range(n)]
        return values,kinds,parameters,agent_parameters,groups,starti
    def get_function_parameters(changes): # incorporate changes to the default parameter values for each given group
        p = dict(jump=0,constant=0,start=0,duration=100,amplitude=1,wavelength=0.005*2*np.pi) # note that all functions accept the same parameters, even if they don't use them all
        parameters = [{**p,**change} for change in changes]
        return parameters
    def reverse_indices(groups): # reverse group ids (assumed to start at 0), e.g. groups 0,1,2 would become 2,1,0
        m = max(groups)
        return [(m-x)%(m+1) for x in groups] # this allows e.g. tendency groups to be reversed in accordance with flipping radicalisation
        
class analysis():
    def isint(a): # check if all elements of anything (i.e. value, list, array, etc) are any kind of integer (e.g. int, int32, int64, float with no significant figures after decimal point, etc)
        def ii(x):
            if isinstance(x,str): # only interested in numerical values (e.g. '1' is not treated as 1)
                return False
            else:
                return np.equal(np.mod(x, 1), 0) # i.e. there are no decimal places, so integer, no matter what the data type is
        s = np.shape(a)
        if s: # i.e. have a list or array of any dimension (if a is scalar, s will be ())
            l = list(map(ii,a))
            if len(np.shape(l))>1: # i.e. have an array rather than a list
                l = sum([list(x) for x in l],[]) # flat list of booleans
        else: # i.e. have a scalar value
            l = [ii(a)]
        return all(l) # this will return True if everything in a is effectively an integer, else False
    def get_neighbour_values(connection_matrix, values): # get neighbouring values for all agents ('neighbours' are agents connected according to the connection matrix, e.g. adjacency matrix at a given time, or sent/received matrix over a given time period)
        def get_neighbours(i,n): # neighbours are agents j that have influenced i
            neighbours = []
            weights = []
            for j in range(n):
                weight = connection_matrix[i,j]
                if weight>0: # although using a weighted average, it's more efficient to exclude unconnected agents
                    neighbours.append(j)
                    weights.append(weight)
            return neighbours,weights
        values_arr = np.array(values)
        n = len(connection_matrix)
        neighbour_values = []
        for i in range(n):
            neighbours,weights = get_neighbours(i,n) # indices and connection strength of neighbours
            if len(neighbours)>0:
                neighbour_value = np.average(values_arr[neighbours],weights=weights)
                neighbour_values.append(neighbour_value)
            else:
                neighbour_values.append(np.nan) # record that there are no neighbours and hence no numerical value
        return neighbour_values
    def get_fractal_dimension(dynamics):
        _,n = np.shape(dynamics)
        fractal_dimensions = []
        for i in range(n):
            pass#fractal_dimensions.append(hfd(dynamics[:,i],opt=False)) # higuchi fractal dimension for time series
        return fractal_dimensions
    def get_autocorrelation(dynamics):
        df = pd.DataFrame(dynamics)
        columns = df.columns
        shape = df.shape
        ac = np.zeros(shape)
        for i,column in enumerate(columns):
            for lag in range(shape[0]-1): # n.b. lag 0 always returns 1 (as correlating with unshifted self), lag n-1 always returns +/-1 (as correlating pairs of numbers which either have the same direction or the opposite)) and lag n always returns nan (as correlating with nothing)
                ac[lag,i] = df[column].autocorr(lag=lag) # lag shifts all data along by the stated lag and fills with nan (compare with manual calculation of correlation if you're confused)
            ac[lag+1,i] = np.nan # avoids divide by 0 warning for lag = n
        return ac
      
class tension():
    def get_tie(i,j,opinions,enemy_threshold=-1): # determine whether agents are friends or enemies, either based on opinion sign (i.e. enemy_threshold=-1), opinion proximity (i.e. enemy_threshold=scalar) or group membership (i.e. enemy_threshold=group list)
        if not isinstance(enemy_threshold,list):
            if enemy_threshold==-1: # determine tie based on sign of opinions, rather than opinion proximity
                tie = np.sign(opinions[i]*opinions[j]) # 1 if both have same sign, else -1
            else: # determine tie based on proximity of opinions
                if abs(opinions[i]-opinions[j]) < enemy_threshold:
                    tie = 1 # i.e. friends
                else:
                    tie = -1 # i.e. enemies
        else: # 'enemy_threshold' is in fact a list of group indices, so judge ties based on these, rather than opinion proximity/sign
            if enemy_threshold[i]==enemy_threshold[j]: # agents are in same group
                tie = 1
            else:
                tie = -1
        return tie
    def get_tie_matrix(adjacency_matrix,opinions,enemy_threshold=-1):
        n = len(adjacency_matrix)
        tie_matrix = np.zeros(shape=(n,n))
        for i in range(n-1): # this nested loop ensures i,j is only considered once (ties are symmetrical, so neglect j,i)
            for j in range(i+1,n):
                if adjacency_matrix[i,j]!=0:
                    tie_matrix[i,j] = tension.get_tie(i,j,opinions,enemy_threshold)
        return tie_matrix
    def get_dyadic_sum(tie_matrix,opinions,enemy_threshold=-1): # used by get_tension() for first term in Eq. 2.1 (see below)
        n = len(tie_matrix)
        dyadic_sum = 0
        for i in range(n-1): # this nested loop ensures i,j is only considered once (ties are symmetrical, so neglect j,i); if i and j not connected, the tie_matrix value will be 0, so don't need to worry about this in the for loop
            for j in range(i+1,n):
                dyadic_sum+=tie_matrix[i,j]*tension.get_tie(i,j,opinions,enemy_threshold)# the paper multiplied the binary opinions unnecessarily (I think), this is still unnecessary if the tie_matrix is based on enemy_threshold, but not if it's based on group membership, so I've included it here
        return dyadic_sum
    def get_triadic_sum(tie_matrix): # used by get_tension() for second term in Eq. 2.1 (see below)
        n = len(tie_matrix)
        triadic_sum = 0
        for i in range(n-2): # this nested loop ensures i,j,k is only considered once (ties are symmetrical, so neglect other permutations of i,j,k in triad)
            for j in range(i+1,n-1):
                if tie_matrix[i,j]!=0: # if statement not strictly necessary, but saves pointless loops
                    for k in range(j+1,n):
                        if tie_matrix[i,k]!=0 and tie_matrix[j,k]!=0: # i.e. i,j,k form a triad (again, if statement not strictly necessary)
                            triadic_sum+=tie_matrix[i,j]*tie_matrix[i,k]*tie_matrix[j,k]
        return triadic_sum
    def get_tension(adjacency_matrix,opinions,enemy_threshold,tension_balance,group_lists=[],group_descriptions=[]): # get tension between all agents at current time point
        # Eq. 2.1 from Minh Pham et al. (2020) http://dx.doi.org/10.1098/rsif.2020.0752  
        tensions = {} # this dictionary will contain dyad, triad and combined tension for proximity/sign- and group-calculated tension at the current time step
        for g,d in zip([enemy_threshold]+group_lists,['proximity']+group_descriptions): # note that 'proximity' could also be 'sign' if enemy_threshold = -1   
            tie_matrix = tension.get_tie_matrix(adjacency_matrix,opinions,g)
            dyadic_sum = tension.get_dyadic_sum(tie_matrix,opinions,enemy_threshold)
            triadic_sum = tension.get_triadic_sum(tie_matrix)
            tension_hamiltonian = -dyadic_sum -tension_balance*triadic_sum
            tensions[d] = [-dyadic_sum, -tension_balance*triadic_sum, tension_hamiltonian]
        return tensions

class plot():
    def add_subplot_label(ax,spl): # add label spl inside brackets and in bold to top left of axes ax
        xl = ax.get_xlim(); yl = ax.get_ylim()
        ax.text(xl[0]-0.1*(xl[1]-xl[0]), yl[1]+0.08*(yl[1]-yl[0]),'('+spl+')', weight='bold') # add subplot label
        return ax
    def get_colours(values,cmap):
        # convert values to colours, see https://stackoverflow.com/questions/28752727/map-values-to-colors-in-matplotlib
        norm = mpc.Normalize(min(values), max(values), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        colours = [mapper.to_rgba(v) for v in values]
        return mapper,colours
    def plot_dynamics(dynamics, properties, variable_name='opinion', description='', cmap='viridis', opacity=0.5, kinds=[], parameters=[], groups=[], starti=0, group_description='', group_cmap='crest', highlight_agent=-1, spl='', tag='', autocorr=False, folder=''): # spl is subplot label to apply (need to join subplots together afterwards)
        # plot opinion time series for each agent with lines coloured according to colourby agent property values
        values = properties[description] # i.e. different properties by which to colour agent trajectories
        n = len(values) # number of agents
        log = False
        if description=='Message Rate':
            values = np.log10(values)
            log = True
        fig = plt.figure(figsize=[6,4]); ax = fig.add_subplot()
        if description=='Tendency Group' or description=='Activity Group':
            cmap = group_cmap # i.e. when plotting opinion (directly related to tendency) or message rate (directly related to activity) on the y-axis, make the colourmap match the group colourmap so it's more intuitive
        mapper,colours = plot.get_colours(values,cmap)
        if len(groups)>0: _,group_colours = plot.get_colours(range(max(groups)+1),group_cmap) # +1 to construct all group indices from maximum index, due to how range works
        # now plot dynamics
        tN = len(dynamics) # number of time points
        t0 = 0; t1 = 0
        if autocorr:
            t1 = 1 # don't plot last lag point (as always nan)
        times = range(t0,tN-t1) # n.b. if plotting autocorrelation, this will be lags instead of times; t1 is the number of 'time' points to drop at end
        for i in range(n): # i.e. plot each agent individually using colour defined by given property
            if i!=highlight_agent:
                colour = mpc.to_rgba(colours[i],opacity)
                zorder = i
            else: # use black line to highlight agent identified by highligh_agent; n.b. default value for highlight_agent is -1, meaning no agent is highlighted
                colour = 'k'
                zorder = n
            ax.plot(times, dynamics[t0:tN-t1,i], color=colour, linestyle='-', linewidth=0.25) # each agent is a column, with time points as rows
        if kinds:
            for i,(kind,parameter) in enumerate(zip(kinds,parameters)):
                colour = mpc.to_rgba(group_colours[starti+i],opacity) # +1 to be consistent with the group indices, assuming 1 base group before the variant groups
                ax.plot(times,[time.time_dependent_function(t,kind=kind,parameters=parameter) for t in times],color=colour,linestyle='-',linewidth=1,label=group_description+' Group %d'%(starti+i))
            if len(kinds)>0:
                ax.legend()
        ax.plot([times[0],times[-1]],[0,0],color='k',linestyle='dashed',linewidth=0.25) # plot horizontal line for 0
        xlabel = 'Time'; ylabel = variable_name
        if autocorr:
            xlabel = xlabel+' lag'
            ylabel = ylabel+' autocorrelation'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if spl: ax = plot.add_subplot_label(ax,spl)
        label = description
        if log: label = 'log('+description+')'
        plt.colorbar(mappable=mapper, label=label)
        title = variable_name
        if autocorr:
            title = title+'_autocorrelation'
        else:
            title = title+'_dynamics'
        plt.savefig(os.path.join(folder,title+'_coloured_by_'+file.underscore(description)+tag+'.png'), dpi=200, bbox_inches='tight') # tag optionally used to distinguish replot
        plt.close()
        return
    def plot_distribution(values,description='',folder=''):
        sns.histplot(x=values[description])
        plt.xlabel(description)
        plt.savefig(os.path.join(folder,'histogram_'+file.underscore(description)+'.png'), dpi=200, bbox_inches='tight')
        plt.close()
        return
    def make_colourbar(values,description='',cmap='viridis',nticks=5,dp=2,filename=''):
        # create standalone colorbar and save to file
        fig = plt.figure()
        mi,ma = [min(values),max(values)]
        left, bottom, width, height = [0, 0., 1., 0.1] # dimensions for axes object
        ax = fig.add_axes([left, bottom, width, height])
        mpl.colorbar.ColorbarBase(ax, orientation='horizontal', cmap=cmap)
        dummy_ticks = list(np.linspace(left,left+width,nticks))
        ticks = list(np.linspace(mi,ma,nticks))
        fmt=fmt='%.'+str(int(dp))+'f'
        ax.set_xticks(dummy_ticks, [fmt%x for x in ticks])
        plt.title(description)
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        return
    '''
    commented out as network diagrams not needed
    def plot_network(connection_matrix, node_values, description='', cmap='viridis', nticks=5, folder=''):
        # use networkx to create network with node size and colour defined by node values, and edge positions and weights defined by connection_matrix
        values = node_values[description]
        if analysis.isint(values): # i.e. dealing with counts, so likely to span orders of magnitude
            values=np.log(values)
            dp = 0
        else:
            dp = 2
        gx = nx.MultiDiGraph() # networkx graph
        mapper,colours = plot.get_colours(values,cmap)
        m,n = np.shape(connection_matrix)
        for i in range(m):
            for j in range(n):
                weight = connection_matrix[i,j]
                if weight>0:
                    if analysis.isint(connection_matrix): # i.e. connection matrix contains counts, so likely to span orders of magnitude
                        weight = np.log(weight)
                    gx.add_edge(str(j),str(i),weight=weight)
        atts = []
        nodes = list(gx.nodes()) # list of all node ids
        for i in nodes: # note that nodes aren't in the same order as node ids, but this doesn't matter, since the node ids correspond to the agent index
            atts.append(dict(color=mpc.to_hex(colours[int(i)]),size=values[int(i)])) # need to take int(i) as the node ids are strings of the integer values
        attrs = dict(zip(nodes,atts))
        nx.set_node_attributes(gx,attrs)
        # plot using pyvis
        gp = Network(notebook=True,directed=True)
        gp.from_nx(gx)
        gp.repulsion(node_distance=100, spring_length=100)
        gp.show_buttons(filter_=['physics'])
        tempname='temp.html'
        gp.show(tempname)
        # make standalone colourbar as image file
        cfilename = os.path.join(folder, 'network_colourbar'+file.underscore(description)+'.png')
        title = description
        if dp==0: title+='(log scale)'
        plot.make_colourbar(values,description=title,cmap=cmap,nticks=nticks,dp=dp,filename=cfilename)
        # add colourbar to html file
        filename = os.path.join(folder, 'network_'+file.underscore(description)+'.html')
        with open(tempname,'r') as read_f, open(filename,'w') as write_f:
            write_f.write('<img src="'+os.path.basename(cfilename)+'">'+'\n')
            for line in read_f:
                write_f.write(line)
        os.remove(tempname)
        return
    '''
    def plot_joint(values1, values2, description1='', description2='', folder=''):
        ax = sns.jointplot(x=values1, y=values2, kind='kde', fill=True) # jointplot doesn't seem to work with a dataframe, so use x and y directly
        ax.set_axis_labels(description1, description2)
        plt.savefig(os.path.join(folder,'jointplot_'+file.underscore(description1)+'_'+file.underscore(description2)+'.png'), dpi=200, bbox_inches='tight')
        plt.close()
        return
    def plot_heatmap(matrix, description='', xlabel='', ylabel='', cmap='viridis', folder=''):
        fig = plt.figure(figsize=[6,4]); ax = fig.add_subplot()
        sns.heatmap(matrix, cmap=cmap, cbar_kws={'label': description}, ax=ax)
        plt.title(description)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(os.path.join(folder,'heatmap_'+file.underscore(description)+'.png'), dpi=200, bbox_inches='tight')
        plt.close()
        return
    def plot_bars(properties, description='', folder=''):
        fig = plt.figure(figsize=[6,4]); ax = fig.add_subplot()
        values = properties[description]
        xs = range(len(values))
        colours=sns.color_palette('colorblind',len(xs))
        ax.bar(xs,values,color=colours)
        plt.xlabel('agent id')
        plt.ylabel(description)
        plt.savefig(os.path.join(folder,'bars_'+file.underscore(description)+'.png'), dpi=200, bbox_inches='tight')
        plt.close()
        return
    def plot_timeseries(tss, types, description, title, labels, xlabel='time', folder=''):
        # tss is a list of ts, which contains list of values at each x-value, e.g. types of tension at each time point
        fig = plt.figure(figsize=[6,4]); ax = fig.add_subplot()
        nl=len(labels) # number of line types to plot
        times = range(len(tss[0])) # time points
        colours = sns.color_palette('colorblind',len(types))
        linestyles = ['dotted','dashed','solid']
        for j in range(nl):
            for i,ts in enumerate(tss):
                ax.plot(times,[t[j] for t in ts],linestyle=linestyles[j],label=types[i]+' '+labels[j],c=colours[i])
        plt.xlabel(xlabel)
        plt.ylabel(description)
        plt.title(title)
        plt.legend(bbox_to_anchor=(1,1),loc='upper left')
        plt.savefig(os.path.join(folder,'timeseries_'+file.underscore(description)+'.png'), dpi=200, bbox_inches='tight')
        plt.close()
        return
               
if __name__=='__main__':
    # function testing
    ii = False # True to test analysis.isint()
    dd = False # True to test scale-free distribution
    pp = False # True to test dyad and triad permulations
    ff = False # True to test time-dependent activity function
    nn = False # True to test neighbouring values
    gg = False # True to test get group from values
    hh = False # test make_folder
    aa = False # test autocorrelation function
    if ii: # test isint() with lists and arrays
        a = [1,2,32,334]
        b = [[1,2,3.0,334],[34,5,3,56],[34,34,2,4]]
        print(analysis.isint(a))
        print(analysis.isint(b))
    if dd: # test scale-free distribution with different parameter values
        fig = plt.figure(figsize=[6,4]); ax = fig.add_subplot()
        colours = ['red','blue','red','blue']
        lss=['-','-',':',':']
        for i,(g,mi) in enumerate(zip([2.1,2.1,1.1,1.1],[0.01,0.1,0.01,0.1])):
            xs = np.linspace(mi,1,100)
            plt.plot(xs,[distribution.pdf(x,g=g,mi=mi) for x in xs],c=colours[i],ls=lss[i],label='$\gamma$=%.1f'%g+', $\epsilon$=%.2f'%mi)
        g=2.1;mi=0.01
        rates=[distribution.scalefree_dist(g=g,mi=mi) for _ in range(1000)]
        plt.hist(rates,density=True,bins=100,label='sampled $\gamma$=%.1f'%g+', $\epsilon$=%.2f'%mi,alpha=0.5)
        plt.legend()
        plt.xlabel('message rate')
        plt.ylabel('probability density')
        #plt.show()
        #plt.savefig('activity_distribution.png', dpi=200, bbox_inches='tight')
    if pp: # test dyad and triad permutations
        def test_perm(n): # number of permutations of dyads and triads with n agents (used to find suitable tension balance for hamiltonian)
            # this demonstrates that:
            # - the number of dyads scales as triangular number of n-1 (https://en.wikipedia.org/wiki/Triangular_number)
            # - the number of triads scales as tetrahedal number of n-2 (https://en.wikipedia.org/wiki/Tetrahedral_number)
            dyad_count=0
            for i in range(n-1):
                for j in range(i+1,n):
                    #print(i,j)
                    dyad_count+=1
            triad_count=0
            for i in range(n-2):
                for j in range(i+1,n-1):
                    for k in range(j+1,n):
                        #print(i,j,k)
                        triad_count+=1
            print('dyad permutations',dyad_count)
            print('triad permutations',triad_count)
            return
        n=100
        print('n', n)
        print('triangular number for n-1', ((n-1)**2+n-1)/2)
        print('tetrahedral number for n-2', (n-2)*(n-1)*n/6)
        print('ratio',((n-1)**2+n-1)/2/((n-2)*(n-1)*n/6))
        test_perm(n)
    if ff: # test time-dependent functions and triggering
        nt=1000
        activity_kinds=['spike']
        activity_parameters=[dict(start=nt+1,amplitude=0,duration=1000)]
        agent_activity_parameters=[dict(start=nt+1,amplitude=0)]
        activity_amp=0.01
        triggers=[1,100,150,500]
        agent_activities_with_time = [] # these are recorded over time for a single agent
        activities = [0] # list of agent's current activities (as a list for consistency with function)
        message_rates = [0.01] # " "
        i=0
        activity_groups=[0]
        for t in range(nt):
            trigger=False
            activity=time.update_activity(i,t,message_rates,activities,activity_kinds,activity_parameters,agent_activity_parameters,activity_groups,activity_amp,trigger)
            agent_activities_with_time.append(activity)
            if t in triggers:
                trigger=True
                message_rates,activities,agent_activity_parameters=time.update_activity(i,t,message_rates,activities,activity_kinds,activity_parameters,agent_activity_parameters,activity_groups,activity_amp,trigger)
                agent_activities_with_time[-1]=activities[0] # [0] as only using 1 agent
        plt.plot(agent_activities_with_time)
        plt.show()
    if nn: # test neighbouring values function
        # this shows that get_neighbour_values works correctly
        # but note that if an agent lacks connections (i.e. receives no messages in the given time frame), no averages can be found (so the function correctly returns np.nan)
        xs = [2.435,1.345,4.345] # agent values
        a = np.array([[0,10,311],[20,0,11],[0,0,0]]) # connection matrix
        ys = analysis.get_neighbour_values(a,xs) # mean neighbour values
        n = len(xs) # number of agents
        print(xs,ys)
        for i,(x,y) in enumerate(zip(xs,ys)):
            if np.isnan(y):
                print('no neighbours for agent '+str(i))
            else:
                # data for neighbours j
                indices = [j for j in range(n) if a[i,j]>0]
                values = [xs[j] for j in range(n) if a[i,j]>0]
                weights = [a[i,j] for j in range(n) if a[i,j]>0]
                mean_values = np.mean(values)
                weighted_average_values = np.average(values,weights=weights)
                print('index='+str(i),', value=%.2f'%x,', function weighted mean neighbour value=%.2f'%y,
                      ', neighbour indices='+str(indices),
                      ', neighbour values='+str(values),
                      ', neighbour weights='+str(weights),
                      ', mean neighbour values=%.2f'%mean_values,
                      ', mean weighted neighbour values=%.2f'%weighted_average_values,
                )
    if gg: # test grouping of existing parameter values
        n=100
        activity_exponent = 2.1 # parameter for activity probability distribution (gamma in PRL paper)
        activity_base = 0.01 # " " (epsilon in PRL paper)
        agent_ids = list(range(n))
        opinion_span = [-1,1]
        message_rate_span = [0.01,1]
        opinions = [random.uniform(*opinion_span) for _ in agent_ids] # randomly assign initial opinions between -1 and 1
        message_rates = [distribution.scalefree_dist(g=activity_exponent, mi=activity_base) for _ in agent_ids]
        opinion_nog = 2
        message_rate_nog = 4
        opinion_groups,opinion_edges = group.get_groups_from_values(opinions,opinion_span,opinion_nog)
        message_rate_groups,message_rate_edges = group.get_groups_from_values(message_rates,message_rate_span,message_rate_nog,log=True)
    if hh:
        file.make_folder(os.path.join('test','this'),'here')
    if aa: # test autocorrelation
        n = 50 # number of agents
        nt = 100 # number of time steps
        nv = 5 # number of discrete property values (assigned at random from 0 to np-1)
        variable_name = 'opinion'
        description = 'prop' # property to use from properties
        dynamics = np.random.rand(nv,n) # make up some agent timeseries data, e.g. opinions
        ac = analysis.get_autocorrelation(dynamics) # get autocorrelation of these
        properties = {'prop':[random.randint(0,nv-1) for _ in range(n)]}
        plot.plot_dynamics(dynamics=ac,properties=properties,variable_name=variable_name,description=description,autocorr=True)