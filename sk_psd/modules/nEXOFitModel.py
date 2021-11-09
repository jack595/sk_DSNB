import histlite as hl
import pandas as pd
import numpy as np
import copy

from matplotlib import pyplot as plt


class nEXOFitModel:

   #########################################################################
   def __init__( self ):
       
       # Here I'll use lists to ensure that the indices are
       # aligned properly.  
       self.pdfs = []
       self.variable_list = []
       self.full_distribution = None
       self.constraints = []
       self.signal_efficiency_flag = False
       self.background_shape_var_flag = False
       self.initial_variable_list = []
       self.signal_name = None
       self.axis_names = None

   #########################################################################
   def AddPDFsFromDataframe( self, input_df, axis_names, append=False, replace_existing_variables=True ):

       self.df_pdfs = input_df
       self.axis_names = axis_names

       if replace_existing_variables:
           if not append:
              self.pdfs = []
              self.variable_list = []

#            print(f"input_df : {input_df}")
#            input_df :               Group                                          Histogram  TotalExpectedCounts
# 0               Far  Hist(2 bins in [0,2], 280 bins in [700.0,3500....         4.920165e+03
# 1      Vessel_Th232  Hist(2 bins in [0,2], 280 bins in [700.0,3500....         2.977887e+03
# 2       Vessel_U238  Hist(2 bins in [0,2], 280 bins in [700.0,3500....         2.185273e+04
# 3    Internals_U238  Hist(2 bins in [0,2], 280 bins in [700.0,3500....         4.635314e+04
# 4   Internals_Th232  Hist(2 bins in [0,2], 280 bins in [700.0,3500....         1.021549e+04
# 5      FullTPC_Co60  Hist(2 bins in [0,2], 280 bins in [700.0,3500....         2.161884e+02
# 6       FullTPC_K40  Hist(2 bins in [0,2], 280 bins in [700.0,3500....         3.249165e+07

           for index, row in input_df.iterrows():
               if (row['Group']=='Total Sum') | (row['Group']=='Off'):
                      continue
               self.pdfs.append( row['Histogram'] )

               this_variable_dict = {}
               this_variable_dict['Name'] = 'Num_{}'.format( row['Group'] )
               this_variable_dict['Value'] = row['TotalExpectedCounts']
               self.variable_list.append( this_variable_dict )
           for var in self.variable_list:
               var['IsFixed'] = False
               var['FitError'] = None
               var['MinuitInputError'] = np.sqrt(var['Value'])
               var['IsConstrained'] = False
               var['Limits'] = (None,None)

           self.initial_variable_list = copy.deepcopy( self.variable_list )

#          each row got a directory
#            print(f"variable_listL: {self.variable_list}")
# variable_listL: [{'Name': 'Num_Far', 'Value': 4920.16518454649, 'IsFixed': False, 'FitError': None, 'MinuitInputError': 70.14388914614366, 'IsConstrained': False, 'Limits': (None, None)}, {'Name': 'Num_Vessel_Th232', 'Value': 2977.886663621045, 'Is
# Fixed': False, 'FitError': None, 'MinuitInputError': 54.57001615925219, 'IsConstrained': False, 'Limits': (None, None)}, {'Name': 'Num_Vessel_U238', 'Value': 21852.725660932727, 'IsFixed': False, 'FitError': None, 'MinuitInputError': 147.8266743890
# 7202, 'IsConstrained': False, 'Limits': (None, None)}, {'Name': 'Num_Internals_U238', 'Value': 46353.14129120649, 'IsFixed': False, 'FitError': None, 'MinuitInputError': 215.2977967634748, 'IsConstrained': False, 'Limits': (None, None)}, {'Name': '
# Num_Internals_Th232', 'Value': 10215.487388093958, 'IsFixed': False, 'FitError': None, 'MinuitInputError': 101.0716942971372, 'IsConstrained': False, 'Limits': (None, None)}, {'Name': 'Num_FullTPC_Co60', 'Value': 216.18837805932716, 'IsFixed': Fals
# e, 'FitError': None, 'MinuitInputError': 14.703345811730307, 'IsConstrained': False, 'Limits': (None, None)}, {'Name': 'Num_FullTPC_K40', 'Value': 32491648.975450445, 'IsFixed': False, 'FitError': None, 'MinuitInputError': 5700.144645134055, 'IsCon
# strained': False, 'Limits': (None, None)}, {'Name': 'Num_Rn222', 'Value': 9107.267168202485, 'IsFixed': False, 'FitError': None, 'MinuitInputError': 95.43200285125783, 'IsConstrained': False, 'Limits': (None, None)}, {'Name': 'Num_FullLXeBb2n', 'Va
# lue': 27949378.272387594, 'IsFixed': False, 'FitError': None, 'MinuitInputError': 5286.717154566489, 'IsConstrained': False, 'Limits': (None, None)}, {'Name': 'Num_FullLXeBb0n', 'Value': 0.0001, 'IsFixed': False, 'FitError': None, 'MinuitInputError
# ': 0.01, 'IsConstrained': False, 'Limits': (None, None)}, {'Name': 'Num_Xe137', 'Value': 46.51665348579301, 'IsFixed': False, 'FitError': None, 'MinuitInputError': 6.8203118320054115, 'IsConstrained': False, 'Limits': (None, None)}]

       else:
           for index, row in input_df.iterrows():
               if (row['Group']=='Total Sum') | (row['Group']=='Off'):
                   continue
               var_idx = self.GetVariableIndexByName( 'Num_{}'.format(row['Group']) )
               self.pdfs[var_idx] = row['Histogram'] 
               self.variable_list[var_idx]['Value'] = row['TotalExpectedCounts'] 

   #########################################################################
   def GenerateModelDistribution( self, fast=False ):
       if len(self.pdfs)==0:
           print('ERROR: no pdfs are loaded into the fitted model.')
           return 


#        print(f"self.pdfs: {self.pdfs}")
#        self.pdfs: [Hist(2 bins in [0,2], 280 bins in [700.0,3500.0], 25 bins in [0.0,650.0], with sum 1.0, 10687 empty bins, and 0 non-finite values), Hist(2 bins in [0,2], 280 bins in [700.0,3500.0], 25 bins in [0.0,650.0], with sum 1.0, 6371 empty bins,
#  and 0 non-finite values), Hist(2 bins in [0,2], 280 bins in [700.0,3500.0], 25 bins in [0.0,650.0], with sum 1.0, 6971 empty bins, and 0 non-finite values), Hist(2 bins in [0,2], 280 bins in [700.0,3500.0], 25 bins in [0.0,650.0], with sum 1.0, 77
# 09 empty bins, and 0 non-finite values), Hist(2 bins in [0,2], 280 bins in [700.0,3500.0], 25 bins in [0.0,650.0], with sum 1.0, 7582 empty bins, and 0 non-finite values), Hist(2 bins in [0,2], 280 bins in [700.0,3500.0], 25 bins in [0.0,650.0], wi
# th sum 1.0, 11577 empty bins, and 0 non-finite values), Hist(2 bins in [0,2], 280 bins in [700.0,3500.0], 25 bins in [0.0,650.0], with sum 1.0, 12359 empty bins, and 0 non-finite values), Hist(2 bins in [0,2], 280 bins in [700.0,3500.0], 25 bins in
#  [0.0,650.0], with sum 1.0, 3606 empty bins, and 0 non-finite values), Hist(2 bins in [0,2], 280 bins in [700.0,3500.0], 25 bins in [0.0,650.0], with sum 1.0, 6353 empty bins, and 0 non-finite values), Hist(2 bins in [0,2], 280 bins in [700.0,3500.
# 0], 25 bins in [0.0,650.0], with sum 1.0, 9559 empty bins, and 0 non-finite values), Hist(2 bins in [0,2], 280 bins in [700.0,3500.0], 25 bins in [0.0,650.0], with sum 1.0, 2018 empty bins, and 0 non-finite values)]

#        print(f"self.variables_list: {self.variable_list}")
#        self.variables_list: [{'Name': 'Num_Far', 'Value': 4920.16518454649, 'IsFixed': False, 'FitError': None, 'MinuitInputError': 70.14388914614366, 'IsConstrained': False, 'Limits': (None, None)}, {'Name': 'Num_Vessel_Th232', 'Value': 2977.886663621045
# , 'IsFixed': False, 'FitError': None, 'MinuitInputError': 54.57001615925219, 'IsConstrained': False, 'Limits': (None, None)}, {'Name': 'Num_Vessel_U238', 'Value': 21852.725660932727, 'IsFixed': False, 'FitError': None, 'MinuitInputError': 147.82667
# 438907202, 'IsConstrained': False, 'Limits': (None, None)}, {'Name': 'Num_Internals_U238', 'Value': 46353.14129120649, 'IsFixed': False, 'FitError': None, 'MinuitInputError': 215.2977967634748, 'IsConstrained': False, 'Limits': (None, None)}, {'Nam
# e': 'Num_Internals_Th232', 'Value': 10215.487388093958, 'IsFixed': False, 'FitError': None, 'MinuitInputError': 101.0716942971372, 'IsConstrained': False, 'Limits': (None, None)}, {'Name': 'Num_FullTPC_Co60', 'Value': 216.18837805932716, 'IsFixed':
#  False, 'FitError': None, 'MinuitInputError': 14.703345811730307, 'IsConstrained': False, 'Limits': (None, None)}, {'Name': 'Num_FullTPC_K40', 'Value': 32491648.975450445, 'IsFixed': False, 'FitError': None, 'MinuitInputError': 5700.144645134055, '
# IsConstrained': False, 'Limits': (None, None)}, {'Name': 'Num_Rn222', 'Value': 9107.267168202485, 'IsFixed': False, 'FitError': None, 'MinuitInputError': 95.43200285125783, 'IsConstrained': False, 'Limits': (None, None)}, {'Name': 'Num_FullLXeBb2n'
# , 'Value': 27949378.272387594, 'IsFixed': False, 'FitError': None, 'MinuitInputError': 5286.717154566489, 'IsConstrained': False, 'Limits': (None, None)}, {'Name': 'Num_FullLXeBb0n', 'Value': 0.0001, 'IsFixed': False, 'FitError': None, 'MinuitInput
# Error': 0.01, 'IsConstrained': False, 'Limits': (None, None)}, {'Name': 'Num_Xe137', 'Value': 46.51665348579301, 'IsFixed': False, 'FitError': None, 'MinuitInputError': 6.8203118320054115, 'IsConstrained': False, 'Limits': (None, None)}]

       # Loop over pdfs, weight them, and sum them together.
       distributions = [
           (self.pdfs[i].values if fast else self.pdfs[i])
            * self.variable_list[i]['Value']
           for i in range(len(self.pdfs))
       ]

       # If we're running with the efficiency parameter, use it to scale the signal
       if self.signal_efficiency_flag:
          sig_idx = self.GetVariableIndexByName( 'Bb0n' )
          sig_eff_idx = self.GetVariableIndexByName('Signal_Efficiency')
          distributions[sig_idx] = distributions[sig_idx] * self.variable_list[sig_eff_idx]['Value']

       if self.background_shape_var_flag:
          sig_idx = self.GetVariableIndexByName( 'Bb0n' )
          bkg_shape_idx = self.GetVariableIndexByName('Background_Shape_Error')
          if fast:
             distributions[sig_idx] = distributions[sig_idx] + \
                                      self.variable_list[bkg_shape_idx]['Value'] * self.pdfs[sig_idx].values
          else:
             distributions[sig_idx] = distributions[sig_idx] + \
                                      self.variable_list[bkg_shape_idx]['Value'] * self.pdfs[sig_idx]

#        print(f"distribution : {distributions}")
#        distribution : [Hist(2 bins in [0,2], 280 bins in [700.0,3500.0], 25 bins in [0.0,650.0], with sum 4920.16518454649, 10687 empty bins, and 0 non-finite values), Hist(2 bins in [0,2], 280 bins in [700.0,3500.0], 25 bins in [0.0,650.0], with sum 2977
# .886663621045, 6371 empty bins, and 0 non-finite values), Hist(2 bins in [0,2], 280 bins in [700.0,3500.0], 25 bins in [0.0,650.0], with sum 21852.725660932727, 6971 empty bins, and 0 non-finite values), Hist(2 bins in [0,2], 280 bins in [700.0,350
# 0.0], 25 bins in [0.0,650.0], with sum 46353.141291206484, 7709 empty bins, and 0 non-finite values), Hist(2 bins in [0,2], 280 bins in [700.0,3500.0], 25 bins in [0.0,650.0], with sum 10215.48738809396, 7582 empty bins, and 0 non-finite values), H
# ist(2 bins in [0,2], 280 bins in [700.0,3500.0], 25 bins in [0.0,650.0], with sum 216.18837805932716, 11577 empty bins, and 0 non-finite values), Hist(2 bins in [0,2], 280 bins in [700.0,3500.0], 25 bins in [0.0,650.0], with sum 32491648.97545044,
# 12359 empty bins, and 0 non-finite values), Hist(2 bins in [0,2], 280 bins in [700.0,3500.0], 25 bins in [0.0,650.0], with sum 9107.267168202485, 3606 empty bins, and 0 non-finite values), Hist(2 bins in [0,2], 280 bins in [700.0,3500.0], 25 bins i
# n [0.0,650.0], with sum 27949378.272387594, 6353 empty bins, and 0 non-finite values), Hist(2 bins in [0,2], 280 bins in [700.0,3500.0], 25 bins in [0.0,650.0], with sum 9.999999999999999e-05, 9559 empty bins, and 0 non-finite values), Hist(2 bins
# in [0,2], 280 bins in [700.0,3500.0], 25 bins in [0.0,650.0], with sum 46.516653485793015, 2018 empty bins, and 0 non-finite values)]

       self.full_distribution = np.sum(distributions, axis=0)
       # print(self.pdfs[0].values)
       # exit(0)
       # print(f"full_distribution: {self.full_distribution}")
       # full_distribution: Hist(2 bins in [0,2], 280 bins in [700.0,3500.0], 25 bins in [0.0,650.0], with sum 60536716.62632619, 1918 empty bins, and 0 non-finite values)
       return self.full_distribution


   #########################################################################
   def IncludeSignalEfficiencyVariableInFit( self, include_flag=True ):

       self.signal_efficiency_flag = include_flag

       # GetVariableIndex returns 1000 if the variable is not found 
       # in the variable_list
       try: 
          self.GetVariableIndexByName('Signal_Efficiency')
          # If we need to turn off the signal efficiency parameter, 
          # delete it from the variable list
          if not self.signal_efficiency_flag:
             sig_eff_idx = self.GetVariableIndexByName('Signal_Efficiency')
             del self.variable_list[ sig_eff_idx ]
       except ValueError:
          # If we need to turn on the signal efficiency parameter,
          # add it to the variable list.
          if self.signal_efficiency_flag:
             this_variable_dict = {}
             this_variable_dict['Name'] = 'Signal_Efficiency'
             this_variable_dict['Value'] = 1.
             this_variable_dict['IsFixed'] = False
             this_variable_dict['IsConstrained'] = False
             this_variable_dict['Limits'] = (None,None)
             this_variable_dict['MinuitInputError'] = 0.01
             this_variable_dict['FitError'] = None
             self.variable_list.append(this_variable_dict)
             self.initial_variable_list.append(this_variable_dict)

   #########################################################################
   def IncludeBackgroundShapeVariableInFit( self, include_flag=True ):

       self.background_shape_var_flag = include_flag

       # GetVariableIndex returns 1000 if the variable is not found 
       # in the variable_list
       try: 
          self.GetVariableIndexByName('Background_Shape_Error')
          # If we need to turn off the signal efficiency parameter, 
          # delete it from the variable list
          if not self.background_shape_var_flag:
             sig_eff_idx = self.GetVariableIndexByName('Background_Shape_Error')
             del self.variable_list[ sig_eff_idx ]
       except ValueError:
          # If we need to turn on the signal efficiency parameter,
          # add it to the variable list.
          if self.background_shape_var_flag:
             this_variable_dict = {}
             this_variable_dict['Name'] = 'Background_Shape_Error'
             this_variable_dict['Value'] = 0.
             this_variable_dict['IsFixed'] = False
             this_variable_dict['IsConstrained'] = False
             this_variable_dict['Limits'] = (None,None)
             this_variable_dict['MinuitInputError'] = 0.01
             this_variable_dict['FitError'] = None
             self.variable_list.append(this_variable_dict)
             self.initial_variable_list.append(this_variable_dict)

   #########################################################################
   def GetVariableIndexByName( self, name ):
       index = -10000
       counter = 0
       for i in range(0,len(self.variable_list)):
           if name in self.variable_list[i]['Name']:
              index = i
              counter += 1
       if index == -10000:
           #print('\n\nERROR: No variable in the likelihood.variable_list contains the name {}.\n'.format(var_name))
           #print('       Please double-check that the variable names in the ComponentsTable match')
           #print('       the ones you\'re trying to access\n\n')
           err_message =  'No variable in the likelihood.variable_list contains the name {}.\n\n'.format(name)
           err_message += '\tPlease double-check that the variable names in the ComponentsTable match\n'
           err_message += '\tthe ones you\'re trying to access\n\n'
           raise ValueError(err_message)

       if counter > 1:
          print('WARNING in nEXOFitModel.GetVariableIndexByName():')
          print('\tFound more than one variable matching the string {}'.format(name))

       return index

   #########################################################################
   def GenerateDataset( self ):
       if self.full_distribution is None:
          self.GenerateModelDistribution()
 
       negative_mask = self.full_distribution.values < 0.
       nonnegative_distribution = self.full_distribution.values
       nonnegative_distribution[negative_mask] = \
                      np.zeros( nonnegative_distribution[negative_mask].shape )
       # Generates a Poisson sample of each bin
       fake_data_values = np.random.poisson( nonnegative_distribution )   
       # Creates a Histlite histogram with the sampled values and errors     
       self.dataset = hl.Hist( self.full_distribution.bins,\
                                 fake_data_values,\
                                 np.sqrt(fake_data_values) )
       return self.dataset


   #########################################################################
   def UpdateVariables( self, variable_numpy_array ):

       if not (len(variable_numpy_array)==len(self.variable_list)):
          print('WARNING: Variable array passed to the model object '+\
                'does not match the length of the variable list. '+\
                'Variables will not be updated.\n')     

       for i in range(0,len(variable_numpy_array)):

          self.variable_list[i]['Value'] = np.copy(variable_numpy_array[i])


   
   #########################################################################
   def GetIntegralInBinRange( self, bin_range_numpy_array ):

       num_dimensions = len(bin_range_numpy_array)       

       # bin_range_numpy_array is an array of arrays, with length
       # num_dimensions. Each sub-array contains the indices over
       # which you want to integrate

       if self.full_distribution is None:
          self.GenerateModelDistribution()

       if num_dimensions != len(self.full_distribution.values.shape):
          print('\nERROR: input array does NOT have the same dimensions as\n')
          print('       this model. Check that your axes are correct.')
          print('       Returning 0.....')
          return 0.
      
       temp_sum = self.full_distribution.values
 
       for i in range(num_dimensions):
           temp_sum = np.sum( temp_sum[ bin_range_numpy_array[i] ], axis=0)
         
       return temp_sum


   #########################################################################
   def GetComponentIntegralInBinRange( self, var_name, bin_range_numpy_array ):

       num_dimensions = len(bin_range_numpy_array)

       var_idx = self.GetVariableIndexByName( var_name )
       this_pdf = self.pdfs[ var_idx ]
        
       if len(bin_range_numpy_array) != len(this_pdf.values.shape):
          print('\nERROR: input array does NOT have the same dimensions as\n')
          print('       this model. Check that your axes are correct.')
          print('       Returning 0.....')
          return 0.

       temp_sum = this_pdf.values * self.variable_list[var_idx]['Value']
       for i in range(num_dimensions):
           temp_sum = np.sum( temp_sum[ bin_range_numpy_array[i] ], axis=0 )

       return temp_sum


   #########################################################################
   def GetSlicedDistribution( self, cut_dict, renormalize = False, \
                              var_name = None, verbose=True ):

       # Check to make sure the cut_dict contains the right axes
       for axis_name in cut_dict.keys():
           if axis_name not in self.axis_names:
              print('\nERROR: \"{}\" is not a valid axis '.format(axis_name) + \
                    'for the PDFs in this model.' )
              print('       Please choose from:')
              for i in range( len(self.axis_names) ):
                  print('          {}'.format(self.axis_names[i]))
              return None
       for axis_name in self.axis_names:
            if axis_name not in list(cut_dict.keys()):
               print('\nERROR: The PDF axis \"{}\"'.format(axis_name) + \
                     ' is not included in the input dict.' )
               print('       Please choose from:')
               for i in range( len(self.axis_names) ):
                   print('          {}'.format(self.axis_names[i]))
               return None
 
       if var_name is not None:
          var_idx = self.GetVariableIndexByName( var_name )
          this_distribution = self.pdfs[var_idx]
       else:
          this_distribution = self.full_distribution
 
       bin_edges = this_distribution.bins
       bin_values = this_distribution.values
 
       new_edges = []
       new_slices = []     
 
       for i in range(len(bin_edges)):
 
            axis_name = self.axis_names[i]
            axis_bins = bin_edges[i]
 
            match_edges_lower_limit = np.where( axis_bins >= cut_dict[axis_name][0] )
            match_edges_upper_limit = np.where( axis_bins <= cut_dict[axis_name][1] )
    
            match_edges = np.intersect1d( match_edges_lower_limit, match_edges_upper_limit )
            match_indices = match_edges[:-1]
  
            new_edges.append( np.array( axis_bins[match_edges] ) )
            new_slices.append( slice(match_indices[0],match_indices[-1]+1) )

            if verbose:
               print('{}:'.format(axis_name))
               print('\tInput cut boundaries:  {:>8.5}, {:>8.5}'.format(\
                      float(cut_dict[axis_name][0]), float(cut_dict[axis_name][1]) ) ) 
               print('\tActual ROI boundaries: {:>8.5}, {:>8.5}'.format(\
                      float(new_edges[i][0]), float(new_edges[i][-1]) ) ) 
   
       sliced_hist = hl.Hist( bins=new_edges, values=bin_values[ tuple(new_slices)  ] )

       if renormalize:
          sliced_hist = sliced_hist.normalize( (0,1,2), integrate=False )

       return sliced_hist







