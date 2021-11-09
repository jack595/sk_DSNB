import numpy as np
import histlite as hl
import nEXOFitModel

import matplotlib.pyplot as plt
from cycler import cycler

from iminuit import Minuit

class nEXOFitLikelihood:

   def __init__( self ):
       # nEXOFitLikelihood is a class that will generate a likelihood function
       # given some arbitrary binned likelihoods and binned PDFs. Both PDFs and
       # datasets are assumed to be in the form of Histlite histograms, and 
       # we assume we're performing a binned maximum likelihood fit.
       self.pdfs = {}
       self.dataset = None
       self.model_distribution = None
       self.best_fit_result = None
       #self.variable_list = []
       self.initial_values = []
       self.model = nEXOFitModel.nEXOFitModel()
       self.nll = np.nan
       self.nll_offset = 0. # -604307041. # This is a legacy number. Offset should be
                                     # recomputed each time you generate a new dataset. 
       self.nll_extra_offset = 0. # This is used to ensure that the minimum is close to zero
                                  # on the second iteration of the fit.
       self.penalize_negative_bins = False
       #self.constraints = []
       self.fitter = None

   ##############################################################################################
   def AddDataset( self, input_dataset ):
       self.dataset = input_dataset
       if self.model_distribution is not None:
          self.nll_offset = 0.
          self.SetInitialOffset()
       self.best_fit_result = None

   ##############################################################################################
   def PenalizeNegativeBins( self, value=True ):
       self.penalize_negative_bins = value

   ##############################################################################################
   def AddPDFDataframeToModel( self, df_pdfs, axis_names, replace_existing_variables=True ):
       self.model.AddPDFsFromDataframe( df_pdfs, \
                                        axis_names, \
                                        replace_existing_variables = replace_existing_variables ) #initialze the fitting parameters list
       self.model_distribution = self.model.GenerateModelDistribution()

   ##############################################################################################
   def ComputeNegLogLikelihood( self, var_values ):
     
       fast = True
       self.model.UpdateVariables( var_values )
       self.model_distribution = self.model.GenerateModelDistribution(fast=fast)
       # print(f"self.model_distribution:{self.model_distribution}")
       # print(f"self.dataset.values: {self.dataset.values}")
       # print(f"nll_offset: {self.nll_offset}")
       # print(f"nll_extra_offset: {self.nll_extra_offset}")
       if not fast:
          self.model_distribution = self.model_distribution.values
      
       # Here we do a binned negative log-likelihood, assuming
       # each bin is an independent, poisson-distributed variable
       mask = (self.model_distribution > 0.)
       self.nll = np.sum( self.model_distribution[mask]  - \
                  self.dataset.values[mask] * np.log( self.model_distribution[mask] ) ) -\
                  self.nll_offset - self.nll_extra_offset 

       if self.penalize_negative_bins:
          # In case we have negative bins, need to penalize the likelihood somehow. For now,
          # I'll just add the sum back to the likelihood...? At least this way the farther 
          # negative it goes, the stronger the penalty.
          mask = (self.model_distribution < 0.)
          self.nll += (-5.)*np.sum(self.model_distribution[mask])

       # Here we apply the constraints (if any) to the variables.
       for constraint in self.model.constraints:
           self.nll += ( var_values[ constraint['Index'] ] - constraint['Value'] )**2 / \
                       ( np.sqrt(2) * constraint['Width']**2 )

       return self.nll

   ##############################################################################################
   def CreateAndRunMinuitFit( self, initial_values=None, print_level=0 ):

       if initial_values is None:
          initial_values = self.GetVariableValues()             

       # print(f"initial_values: {initial_values}")
# initial_values: [4.92016518e+03 2.97788666e+03 2.18527257e+04 4.63531413e+04
#  1.02154874e+04 2.16188378e+02 3.24916490e+07 9.10726717e+03
#  2.79493783e+07 0.00000000e+00 4.65166535e+01]
       self.fitter = Minuit.from_array_func( self.ComputeNegLogLikelihood, \
                                   np.copy(initial_values),\
                                   error = self.GetMinuitInputErrorTuple(), \
                                   fix   = self.GetVariableFixTuple(), \
                                   name  = self.GetVariableNamesTuple(), \
                                   limit = self.GetVariableLimitsTuple(), \
                                   errordef = 0.5,\
                                   print_level = print_level)
       self.fitter.migrad()
       #self.PrintVariableList()
       num_iterations = 1
       while not (self.fitter.get_fmin()['is_valid']\
             and self.fitter.get_fmin()['has_accurate_covar']):    
                 if num_iterations > 9:
                    break
                 if print_level > 0:
                    print('Fit iteration {}'.format(num_iterations+1))

                 #########################################################
                 # Fluctuate the inputs on the second pass, but make sure the
                 # 'fixed' variables stay the same
                 fluctuated_input_values = self.GetVariableValues() + \
                                     np.random.randn(len(initial_values)) * \
                                     np.sqrt( self.GetVariableValuesNonNegative() )
                 for i in range(0,len(self.model.variable_list)):
                          if self.model.variable_list[i]['IsFixed']:
                             fluctuated_input_values[i] = initial_values[i]
                 ##########################################################
          
                 self.fitter = Minuit.from_array_func( self.ComputeNegLogLikelihood, \
                                        np.copy(fluctuated_input_values),\
                                        error = self.GetMinuitInputErrorTuple(), \
                                        fix   = self.GetVariableFixTuple(), \
                                        name  = self.GetVariableNamesTuple(), \
                                        limit = self.GetVariableLimitsTuple(), \
                                        errordef = 0.5,\
                                        print_level = print_level)
                 self.fitter.migrad()
                 num_iterations += 1
                 #self.PrintVariableList()


       #############################################################################
       # If the fit did converge, make a small perturbation and run it again to make
       # sure it finds the absolute minimum.
       print('Attempting to refine the fit...')
       if self.fitter.get_fmin()['is_valid'] \
          and self.fitter.get_fmin()['has_accurate_covar']:
                 fluctuated_input_values = self.GetVariableValues() + \
                                     np.random.randn(len(initial_values)) * \
                                     0.5 * self.fitter.np_errors()
                 for i in range(0,len(self.model.variable_list)):
                          if self.model.variable_list[i]['IsFixed']:
                             fluctuated_input_values[i] = initial_values[i]
                 second_pass_fitter = Minuit.from_array_func( self.ComputeNegLogLikelihood, \
                                        np.copy(fluctuated_input_values),\
                                        error = self.GetMinuitInputErrorTuple(), \
                                        fix   = self.GetVariableFixTuple(), \
                                        name  = self.GetVariableNamesTuple(), \
                                        limit = self.GetVariableLimitsTuple(), \
                                        errordef = 0.5,\
                                        print_level = print_level)
                 second_pass_fitter.migrad()

                 # If the new fit is an improvement, replace the old one 
                 if second_pass_fitter.fval < self.fitter.fval \
                    and second_pass_fitter.get_fmin()['is_valid'] \
                    and second_pass_fitter.get_fmin()['has_accurate_covar']:
                    self.fitter = second_pass_fitter 
       ############################################################################
              
       fit_errors = self.fitter.errors
       for var in self.model.variable_list:
           var['FitError'] = fit_errors[ var['Name'] ]
           

       return self.fitter.get_fmin()['is_valid'],\
              self.fitter.get_fmin()['has_accurate_covar'],\
              num_iterations 

   ##############################################################################################
   def ComputeLambda( self, initial_values=None, signal_name='Bb0n', print_level=0,\
                      signal_expectation=None, fixed_fit_signal_value=None, repeat_best_fit=False):

       

       signal_idx = self.GetVariableIndex( signal_name )

       initial_values_best = np.copy(initial_values)
       initial_values_fixed = np.copy(initial_values)

       if initial_values is not None:

          if signal_expectation is not None:
             initial_values_best[signal_idx] = signal_expectation

          if fixed_fit_signal_value is not None:
             initial_values_fixed[signal_idx] = fixed_fit_signal_value

       # Compute the best fit for if you haven't already (the variable
       # best_fit_result is reset each time a new dataset is added).
       if self.best_fit_result is None or repeat_best_fit:

          self.SetVariableFixStatus( signal_name, False)
          print('\nRunning best fit...\n')
          ################################################################################
          # The action happens here!
          best_fit_converged, best_fit_covar, best_fit_iterations = \
                           self.CreateAndRunMinuitFit( initial_values_best, print_level=print_level )
          best_fit_parameters = dict( self.fitter.values ) 
          best_fit_errors = dict( self.fitter.errors )
  
          self.best_fit_result = dict() 
          self.best_fit_result['best_fit_converged'] = best_fit_converged
          self.best_fit_result['best_fit_covar'] = best_fit_covar
          self.best_fit_result['best_fit_iterations'] = best_fit_iterations
          self.best_fit_result['best_fit_parameters'] = best_fit_parameters
          self.best_fit_result['best_fit_errors'] = best_fit_errors
          self.best_fit_result['nll'] = self.fitter.fval

          if print_level == 1:
             self.PrintVariableList()
   
       print('\n\nFit with {} fixed at {:3.3} cts...\n'.format( self.model.variable_list[signal_idx]['Name'], \
                                                                initial_values_fixed[signal_idx] ) )

       # Fix the signal variable
       self.SetVariableFixStatus( signal_name, True)

       ################################################################################
       # The action happens here!
       fixed_fit_converged, fixed_fit_covar, fixed_fit_iterations = \
                        self.CreateAndRunMinuitFit( initial_values_fixed, print_level=print_level )
       fixed_fit_parameters = dict( self.fitter.values )
       fixed_fit_errors = dict( self.fitter.errors ) 

       # Note, the 2 is positive here becuase fval is the *negative* log-likelihood
       # Lambda is defined as -2 * ln( L_fixed / L_best )
       this_lambda = 2.*(self.fitter.fval - self.best_fit_result['nll'])

       # Unfix the signal variable
       self.SetVariableFixStatus( signal_name, False)

       lambda_result = dict()

       lambda_result['best_fit_nll']         = self.best_fit_result['nll']
       lambda_result['fixed_fit_nll']        = self.fitter.fval
       lambda_result['lambda']               = this_lambda
       lambda_result['best_fit_converged']   = self.best_fit_result['best_fit_converged']
       lambda_result['best_fit_covar']       = self.best_fit_result['best_fit_covar']
       lambda_result['best_fit_iterations']  = self.best_fit_result['best_fit_iterations']
       lambda_result['best_fit_parameters']  = self.best_fit_result['best_fit_parameters']
       lambda_result['best_fit_errors']      = self.best_fit_result['best_fit_errors']
       lambda_result['fixed_fit_converged']  = fixed_fit_converged 
       lambda_result['fixed_fit_covar']      = fixed_fit_covar
       lambda_result['fixed_fit_iterations'] = fixed_fit_iterations 
       lambda_result['fixed_fit_parameters'] = fixed_fit_parameters
       lambda_result['fixed_fit_errors']     = fixed_fit_errors 

       return lambda_result

   ##############################################################################################

   ##############################################################################################
   def ComputeLambdaForPositiveSignal( self, initial_values=None, signal_name='Bb0n',\
                    print_level=0, signal_expectation=None, fixed_fit_signal_value=None, repeat_best_fit=False):
       '''
       Here we calculate the test statistic for the case where the signal is assumed
       to be positive. This means that, if the best fit number of signal counts is
       negative, we regard this as unphysical, and evalute the likelihood ratio with
       the signal fixed to 0 in the denominator (rather than the global best fit). 
 
       This corresponds to Eq. 11 in Cowan et al., "Asymptotic formulae for likelihood-based 
       tests of new physics" (arXiv:1007.1727)
       '''

       signal_idx = self.GetVariableIndex( signal_name )
       initial_values_best = np.copy(initial_values)
       initial_values_fixed = np.copy(initial_values)
#        print(f"signal_index: {signal_idx}")
#        print(f"initial_values_fixed: {initial_values_fixed}")
#        print(f"initial_values_best: {initial_values_best}")
# signal_index: 9
# initial_values_fixed: [4.92016518e+03 2.97788666e+03 2.18527257e+04 4.63531413e+04
#  1.02154874e+04 2.16188378e+02 3.24916490e+07 9.10726717e+03
#  2.79493783e+07 0.00000000e+00 4.65166535e+01]
# initial_values_best: [4.92016518e+03 2.97788666e+03 2.18527257e+04 4.63531413e+04
#  1.02154874e+04 2.16188378e+02 3.24916490e+07 9.10726717e+03
#  2.79493783e+07 0.00000000e+00 4.65166535e+01]

       if initial_values is not None:

          if signal_expectation is not None:
             initial_values_best[signal_idx] = signal_expectation

          if fixed_fit_signal_value is not None:
             initial_values_fixed[signal_idx] = fixed_fit_signal_value

       # Compute the best fit for if you haven't already (the variable
       # best_fit_result is reset to None each time a new dataset is added).
       if self.best_fit_result is None or repeat_best_fit:
            self.SetVariableFixStatus( signal_name, False)
            print('\nRunning best fit...\n')
            best_fit_converged, best_fit_covar, best_fit_iterations = \
                             self.CreateAndRunMinuitFit( initial_values_best, print_level=print_level )
            best_fit_parameters = dict( self.fitter.values ) 
            best_fit_errors = dict( self.fitter.errors )
  
            self.best_fit_result = dict() 
            self.best_fit_result['best_fit_converged'] = best_fit_converged
            self.best_fit_result['best_fit_covar'] = best_fit_covar
            self.best_fit_result['best_fit_iterations'] = best_fit_iterations
            self.best_fit_result['best_fit_parameters'] = best_fit_parameters
            self.best_fit_result['best_fit_errors'] = best_fit_errors
            self.best_fit_result['nll'] = self.fitter.fval
            best_fit_signal_num = [val for key, val in best_fit_parameters.items() if signal_name in key]
            self.best_fit_result['best_fit_signal_counts'] = best_fit_signal_num[0] 
            print('Best fit result NLL: {:4.4}'.format(self.fitter.fval))

            # print(f"best_fit_result: {self.best_fit_result}")
            # best_fit_result: {'best_fit_converged': True, 'best_fit_covar': True, 'best_fit_iterations': 1, 'best_fit_parameters': {'Num_Far': 4190.334988465915, 'Num_Vessel_Th232': 3459.077966780691, 'Num_Vessel_U238': 22856.728321088478, 'Num_Internals_U238'
# : 47602.799514586986, 'Num_Internals_Th232': 9667.498682791736, 'Num_FullTPC_Co60': 216.18837805932716, 'Num_FullTPC_K40': 32482970.02499483, 'Num_Rn222': 8435.782772164863, 'Num_FullLXeBb2n': 27949958.349978372, 'Num_FullLXeBb0n': 1.55922592615098
# 67e-06, 'Num_Xe137': 55.4561253893367}, 'best_fit_errors': {'Num_Far': 1005.2414836107653, 'Num_Vessel_Th232': 417.2256736193167, 'Num_Vessel_U238': 2472.147028433439, 'Num_Internals_U238': 1374.4922708480444, 'Num_Internals_Th232': 415.68135486523
# 26, 'Num_FullTPC_Co60': 14.703345811730307, 'Num_FullTPC_K40': 6125.60478474386, 'Num_Rn222': 839.0884583732272, 'Num_FullLXeBb2n': 5667.915432497859, 'Num_FullLXeBb0n': 2.4952865728383955, 'Num_Xe137': 23.457227125308894}, 'nll': -1.90709490429467
# 54, 'best_fit_signal_counts': 1.5592259261509867e-06}


            if print_level == 1:
               self.PrintVariableList()

            # If the best fit number of signal events is less than 0, re-evaluate lambda
            # with the signal fixed at 0. 
            if len(best_fit_signal_num) > 1:
                 print('WARNING: more than one variable matches the signal name ({})'.format(signal_name))
                 print('THIS WILL CAUSE PROBLEMS\n')
                 raise ValueError
            if self.best_fit_result['best_fit_signal_counts'] < 0.:
                 print('Best fit signal value is less than 0! Fixing value to 0 and re-running fit...')
                 self.SetVariableFixStatus( signal_name, True )
                 initial_values_best[signal_idx] = 0.
                 zero_fit_converged, zero_fit_covar, zero_fit_iterations = \
                                  self.CreateAndRunMinuitFit( initial_values_best, print_level=print_level )
                 zero_fit_parameters = dict( self.fitter.values ) 
                 zero_fit_errors = dict( self.fitter.errors )
                 self.zero_fit_result = dict() 
                 self.zero_fit_result['zero_fit_converged'] = zero_fit_converged
                 self.zero_fit_result['zero_fit_covar'] = zero_fit_covar
                 self.zero_fit_result['zero_fit_iterations'] = zero_fit_iterations
                 self.zero_fit_result['zero_fit_parameters'] = zero_fit_parameters
                 self.zero_fit_result['zero_fit_errors'] = zero_fit_errors
                 self.zero_fit_result['nll'] = self.fitter.fval
                 print('Zero fit result NLL: {:4.4}'.format(self.fitter.fval))       
                 if print_level == 1:
                    self.PrintVariableList()

                 self.SetVariableFixStatus( signal_name, False)
   
       print('\n\nFit with {} fixed at {:3.3} cts...\n'.format( self.model.variable_list[signal_idx]['Name'], \
                                                                initial_values_fixed[signal_idx] ) )

       # Fix the signal variable
       self.SetVariableFixStatus( signal_name, True)

       fixed_fit_converged, fixed_fit_covar, fixed_fit_iterations = \
                        self.CreateAndRunMinuitFit( initial_values_fixed, print_level=print_level )
       fixed_fit_parameters = dict( self.fitter.values )
       fixed_fit_errors = dict( self.fitter.errors ) 

       # Note, the 2 is positive here becuase fval is the *negative* log-likelihood
       # Lambda is defined as -2 * ln( L_fixed / L_best )
       if self.best_fit_result['best_fit_signal_counts'] < 0.:
          print('Zero fit result NLL: {:4.4}'.format(self.zero_fit_result['nll']))
          this_lambda = 2.*(self.fitter.fval - self.zero_fit_result['nll'])
       else:
          this_lambda = 2.*(self.fitter.fval - self.best_fit_result['nll'])

       # Unfix the signal variable
       self.SetVariableFixStatus( signal_name, False)

       lambda_result = dict()

       lambda_result['lambda']               = this_lambda
       lambda_result['best_fit_converged']   = self.best_fit_result['best_fit_converged']
       lambda_result['best_fit_covar']       = self.best_fit_result['best_fit_covar']
       lambda_result['best_fit_iterations']  = self.best_fit_result['best_fit_iterations']
       lambda_result['best_fit_parameters']  = self.best_fit_result['best_fit_parameters']
       lambda_result['best_fit_errors']      = self.best_fit_result['best_fit_errors']
       lambda_result['fixed_fit_converged']  = fixed_fit_converged 
       lambda_result['fixed_fit_covar']      = fixed_fit_covar
       lambda_result['fixed_fit_iterations'] = fixed_fit_iterations 
       lambda_result['fixed_fit_parameters'] = fixed_fit_parameters
       lambda_result['fixed_fit_errors']     = fixed_fit_errors 

       return lambda_result

   ##############################################################################################

   ##############################################################################################
   def GetVariableValues( self ):
       # Returns a numpy array with the variable values
       array = np.zeros(len(self.model.variable_list))
       for i in range(0,len(self.model.variable_list)):
           array[i] = np.copy(self.model.variable_list[i]['Value'])
       return array

   ##############################################################################################
   def GetVariableValuesNonNegative( self ):
       # Returns a numpy array with the variable values
       array = np.zeros(len(self.model.variable_list))
       for i in range(0,len(self.model.variable_list)):
           if self.model.variable_list[i]['Value'] < 0.:
              continue
           else:
              array[i] = np.copy(self.model.variable_list[i]['Value'])
       return array


   ##############################################################################################
   def SetVariableLimits( self, var_name, upper_limit=None, lower_limit=None):
       var_idx = self.GetVariableIndex( var_name )
       self.model.variable_list[var_idx]['Limits'] = (lower_limit,upper_limit)

   ##############################################################################################
   def PrintVariableList( self ):
       print('{:<23} {:<12} {:<9} {:<10} {:<13} {:<14} {:<13}'.format('Variable name:','Value:',\
                                                        'IsFixed:','FitError','InputError:','IsConstrained:','Limits:'))
       for var in self.model.variable_list:
           print('{:<23} {:<12.4} {:<9} {:<10.4} {:<13.4} {:<14} ({:4},{:4})'.format(var['Name'], \
                                                                      var['Value'],\
                                                                  str(var['IsFixed']),\
                                                                  str(var['FitError']),\
                                                                      var['MinuitInputError'],\
                                                                  str(var['IsConstrained']),\
                                                                  str(var['Limits'][0]),str(var['Limits'][1]) ))

   ##############################################################################################
   def GetVariableIndex( self, var_name ):
       return self.model.GetVariableIndexByName( var_name )

   ##############################################################################################
   def SetInitialOffset( self ):
       if self.model.variable_list == []:
          print('ERROR: Attempting to compute the offset, but \n' +\
                '       there are no initial values available.\n' +\
                '       Please add a model before proceeding.')
          return
       else:
          input_vals_array = np.array([var['Value'] for var in self.model.variable_list])
          self.nll_extra_offset = 0.
          self.nll_offset = self.ComputeNegLogLikelihood( input_vals_array )
       return


   ##############################################################################################
   def SetAllVariablesFloating( self ):
       for i in range(len(self.model.variable_list)):
           self.model.variable_list[i]['IsFixed'] = False

   ##############################################################################################
   def SetVariableFixStatus( self, var_name, isFixedInput ):
       var_idx = self.GetVariableIndex( var_name )
       (self.model.variable_list[var_idx])['IsFixed'] = isFixedInput

   ##############################################################################################
   def SetFractionalMinuitInputError( self, var_name, new_minuit_error ):
       # Here, the input should be a fraction (say, 0.05) and we'll
       # scale that to the value of the variable.
       var_idx = self.GetVariableIndex( var_name )
       (self.model.variable_list[var_idx])['MinuitInputError'] =  \
              (self.model.variable_list[var_idx])['Value'] * new_minuit_error

   ##############################################################################################
   def GetVariableFixTuple( self ):
       # iMinuit requires a tuple of booleans to tell it which parameters
       # should be fixed in the fit.
       return tuple( var['IsFixed'] for var in self.model.variable_list )

   ##############################################################################################
   def GetVariableNamesTuple( self ):
       # iMinuit requires a tuple containing the names of each variable
       return tuple( var['Name'] for var in self.model.variable_list )

   ##############################################################################################
   def GetVariableLimitsTuple( self ):
       return tuple( var['Limits'] for var in self.model.variable_list )

   ##############################################################################################
   def GetMinuitInputErrorTuple( self ):
       # iMinuit requires a tuple containing the "error", which
       # is a parameter that I think defines the step size.
       return tuple( var['MinuitInputError'] for var in self.model.variable_list )

   ##############################################################################################
   def SetGaussianConstraintAbsolute( self, var_name, constraint_value, constraint_width ):
       var_idx = self.GetVariableIndex( var_name )
       if not self.model.variable_list[var_idx]['IsConstrained']:
          self.model.variable_list[var_idx]['IsConstrained'] = True
          self.model.constraints.append( {'Name': self.model.variable_list[var_idx]['Name'], \
                                          'Index': var_idx, \
                                          'Value': constraint_value, \
                                          'Width': constraint_width } )
       else:
          for constraint in self.model.constraints:
              if var_name in constraint['Name']:
                 constraint['Value'] = constraint_value
                 constraint['Width'] = constraint_width
                 break
              else:
                 continue                 


   ##############################################################################################
   def SetGaussianConstraintFractional( self, var_name, constraint_value, constraint_width_fractional ):
       var_idx = self.GetVariableIndex( var_name )
       if not self.model.variable_list[var_idx]['IsConstrained']:
          self.model.variable_list[var_idx]['IsConstrained'] = True
          self.model.constraints.append( {'Name': self.model.variable_list[var_idx]['Name'], \
                                          'Index': var_idx, \
                                          'Value': constraint_value, \
                                          'Width': constraint_width_fractional * constraint_value } )
       else:
          for constraint in self.model.constraints:
              if var_name in constraint['Name']:
                 constraint['Value'] = constraint_value
                 constraint['Width'] = constraint_width_fractional *  constraint_value
                 break
              else:
                 continue                 

   ##############################################################################################
   def ClearConstraints( self ):
       self.model.constraints = []
       for var in self.model.variable_list:
           var['IsConstrained'] = False


   ##############################################################################################
   def PlotModelDistributions( self, ss_cut_dict, ms_cut_dict, \
                               output_filename='test_plot.png', plot_data=False, \
                               save=True, show=False ):

       # Set up the plotting parameters
       initial_cycler = plt.rcParams['axes.prop_cycle']
       plt.rcParams.update({'font.size': 14})
       custom_cycler = cycler( color = [ (1.,0.5,0.),\
                                         (0.,0.,1.,0.5),\
                                         (0.,0.8,0.),\
                                         (1.,0.,0.),\
                                         (0.5,1.,0.5),\
                                         (0.,0.8,0.8),\
                                         (0.1,0.6,0.5),\
                                         (1.,0.,1.),\
                                         (0.,0.,0.,0.2),\
                                         (0.4,0.4,0.4),\
                                         (0.5,0.,0.5) ] )
       plt.rc('axes', prop_cycle=custom_cycler)

       self.fig, self.ax = plt.subplots (2, 2, figsize=(15, 10))#fys
       #self.fig, self.ax = plt.subplots (2, 2, figsize=(12, 10))


       # Loop over pdfs and each to plot
       for i in range(len(self.model.variable_list)):
           var = self.model.variable_list[i]
           if 'Num' in var['Name']:

              weight = var['Value']
              ss_pdf = self.model.GetSlicedDistribution( ss_cut_dict, var_name=var['Name'], verbose=False )
              ms_pdf = self.model.GetSlicedDistribution( ms_cut_dict, var_name=var['Name'], verbose=False )
              component_name = ''.join( var['Name'].split('_')[1:] )
              print('Plotting {}'.format(component_name))


              if i == 0:
                 # Initialize the summed histograms      
                 ss_sum = hl.hist( [np.array([0.]),np.array([0.]),np.array([0.])] , \
                                   bins=ss_pdf.bins)
                 ms_sum = hl.hist( [np.array([0.]),np.array([0.]),np.array([0.])] , \
                                   bins=ms_pdf.bins)
              else:
                 ss_sum += ( weight * ss_pdf )
                 ms_sum += ( weight * ms_pdf )

              hl.plot1d( self.ax[0,0], (weight * ss_pdf).project([1]), label=component_name )
              hl.plot1d( self.ax[0,1], (weight * ms_pdf).project([1]) )
              hl.plot1d( self.ax[1,1], (weight * ms_pdf).project([2]) )
              hl.plot1d( self.ax[1,0], (weight * ss_pdf).project([2]) )

              if 'Bb0n' in component_name:
                 hl.fill_between( self.ax[0,0], 0, (weight * ss_pdf).project([1]), color=(0.5,0.5,0.5), alpha=0.1 )
                 hl.fill_between( self.ax[0,1], 0, (weight * ms_pdf).project([1]), color=(0.5,0.5,0.5), alpha=0.1 )
                 hl.fill_between( self.ax[1,1], 0, (weight * ms_pdf).project([2]), color=(0.5,0.5,0.5), alpha=0.1 )
                 hl.fill_between( self.ax[1,0], 0, (weight * ss_pdf).project([2]), color=(0.5,0.5,0.5), alpha=0.1 )

       hl.plot1d(self.ax[0,0],ss_sum.project([1]),color='b',label='Total Sum')
       hl.plot1d(self.ax[0,1],ms_sum.project([1]),color='b')
       hl.plot1d(self.ax[1,1],ms_sum.project([2]),color='b')
       hl.plot1d(self.ax[1,0],ss_sum.project([2]),color='b') 

       if plot_data:
          ss_data = self.GetSlicedDataset( ss_cut_dict, verbose=False ) 
          ms_data = self.GetSlicedDataset( ms_cut_dict, verbose=False ) 

          hl.plot1d( self.ax[0,0], ss_data.project([1]),crosses=False,\
                     label='Toy Data Sample',color='k',linewidth=1.0)
          hl.plot1d( self.ax[0,1], ms_data.project([1]),crosses=False,linewidth=1.0,color='k')
          hl.plot1d( self.ax[1,1], ms_data.project([2]),crosses=False,linewidth=1.0,color='k')
          hl.plot1d( self.ax[1,0], ss_data.project([2]),crosses=False,linewidth=1.0,color='k')

       self.fig.legend(ncol=5,facecolor=(1.,1.,1.),framealpha=1.,loc='upper center')
       self.ax[0,0].set_ylim(1e-2,1e7)
       self.ax[0,0].set_xlim(700.,3500.)
       self.ax[0,0].set_ylabel('Counts')
       self.ax[0,0].set_xlabel('Energy (keV)')
       self.ax[0,0].set_yscale('log')
       self.ax[0,1].set_ylim(1e-2,1e7)
       self.ax[0,1].set_xlim(700.,3500.)
       self.ax[0,1].set_ylabel('Counts')
       self.ax[0,1].set_xlabel('Energy (keV)')
       self.ax[0,1].set_yscale('log')
       self.ax[1,1].set_ylim(1e-2,1e7)
       self.ax[1,1].set_xlim(0.,640.)
       self.ax[1,1].set_ylabel('Counts')
       self.ax[1,1].set_xlabel('Standoff (mm)')
       self.ax[1,1].set_yscale('log')
       self.ax[1,0].set_ylim(1e-2,1e7)
       self.ax[1,0].set_xlim(0.,640.)
       self.ax[1,0].set_ylabel('Counts')
       self.ax[1,0].set_xlabel('Standoff (mm)')
       self.ax[1,0].set_yscale('log')

       if save:
          plt.savefig(output_filename,dpi=200,bbox_inches='tight')

       if show:
          plt.show()

       plt.rc('axes', prop_cycle=initial_cycler)

       return


   #########################################################################
   def GetSlicedDataset( self, cut_dict, renormalize = False, \
                               verbose=True ):

       # Check to make sure the cut_dict contains the right axes
       for axis_name in cut_dict.keys():
           if axis_name not in self.model.axis_names:
              print('\nERROR: \"{}\" is not a valid axis '.format(axis_name) + \
                    'for the PDFs in this model.' )
              print('       Please choose from:')
              for i in range( len(self.model.axis_names) ):
                  print('          {}'.format(self.model.axis_names[i]))
              return None
       for axis_name in self.model.axis_names:
            if axis_name not in list(cut_dict.keys()):
               print('\nERROR: The PDF axis \"{}\"'.format(axis_name) + \
                     ' is not included in the input dict.' )
               print('       Please choose from:')
               for i in range( len(self.model.axis_names) ):
                   print('          {}'.format(self.model.axis_names[i]))
               return None
 

       if self.dataset is not None:
          this_distribution = self.dataset
       else:
          print('\nERROR: no dataset in likelihood object.')
          return None 

       bin_edges = this_distribution.bins
       bin_values = this_distribution.values
 
       new_edges = []
       new_slices = []     
 
       for i in range(len(bin_edges)):
 
            axis_name = self.model.axis_names[i]
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



 
