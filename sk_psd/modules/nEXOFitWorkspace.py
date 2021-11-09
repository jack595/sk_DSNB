import histlite as hl
import pandas as pd
import numpy as np
import sys
import yaml
import time
import uproot
import os
#import iminuit

import nEXOExcelTableReader
import nEXOMaterialsDBInterface

#from MC_ID_Dict_2019Baseline import MC_ID_Dict

class nEXOFitWorkspace:

   ##########################################################################
   # Constructor just initializes empty objects.
   ##########################################################################
   def __init__( self, config='./config/TUTORIAL_config.yaml'):
      
      # Initialize the class members to null objects.
      #     'df_components' will be a pandas dataframe which
      #              contains the individual simulation
      #              PDFs as histlite histograms and the
      #              relevant info from the materials DB
      #              (component mass, activity, etc.)
      #     'df_group_pdfs' will be a pandas dataframe
      #              which contains the distributions for
      #              each group, weighted appropriately
      #     'neg_log_likelihood' will be a customized
      #              object which computes a likelihood
      #              between the grouped PDFs and some toy
      #              dataset
      #     'minimizer' will be an iMinuit object
      #     'signal_counts' allows the user to manually set
      #              the number of 0nuBB events in the 
      #              dataset
      #     'livetime' is the expected livetime of the 
      #              experiment in units of seconds. 
      #              #TOFIX: this will need to be modified 
      #                      for isotopes that have livetimes
      #                      on the order of years or less    
      #     'histogram_axis_names' contains the names of the
      #              variables we're fitting. These should
      #              be already associated with the PDFs at
      #              an earlier stage in the data processing -
      #              the present code should be totally agnostic
      #              to the variables/binning/etc. 

      self.df_components = pd.DataFrame()       
      self.df_group_pdfs = pd.DataFrame()
      self.neg_log_likelihood = None
      self.minimizer = None
      self.signal_counts=0.0001
      #self.livetime = 1. * 365. * 24. * 60. * 60.
      self.livetime = 10. * 365. * 24. * 60. * 60.
      self.histogram_axis_names = None
      self.roi_indices = None
      self.roi_edges = None

      self.materialsDB = None

      self.fluctuate_radioassay_during_grouping = False
      self.radioassay_limits_handling = "Delta"

      config_file = open(config,'r')
      self.config = yaml.load( config_file, Loader=yaml.SafeLoader )
#       print(f"config: {self.config}")
# config: {'SummaryInputFile': '../tables/Summary_D-005_v22_2018-11-12_02_tutorial.h5', 'HistogramAxisNames': ['SS/MS', 'Energy (keV)', 'Standoff (mm)'], 'CustomSpecActivities': {'bb0n_FullLXe': 1e-10}, 'CustomScalingFactors': {'Xe137_ActiveLXe': 1.0
# , 'Xe137_InactiveLXe': 1.0}, 'GroupAssignments': {'U238_OuterCryostatResin': 'Far', 'U238_OuterCryostatFiber': 'Far', 'U238_OuterCryostatSupportResin': 'Far', 'U238_OuterCryostatSupportFiber': 'Far', 'U238_InnerCryostatResin': 'Far', 'U238_InnerCry
# ostatFiber': 'Far', 'U238_InnerCryostatSupportResin': 'Far', 'U238_InnerCryostatSupportFiber': 'Far', 'U238_InnerCryostatLiner': 'Far', 'Th232_OuterCryostatResin': 'Far', 'Th232_OuterCryostatFiber': 'Far', 'Th232_OuterCryostatSupportResin': 'Far',
# 'Th232_OuterCryostatSupportFiber': 'Far', 'Th232_InnerCryostatResin': 'Far', 'Th232_InnerCryostatFiber': 'Far', 'Th232_InnerCryostatSupportResin': 'Far', 'Th232_InnerCryostatSupportFiber': 'Far', 'Th232_InnerCryostatLiner': 'Far', 'Co60_OuterCryost
# atResin': 'Far', 'Co60_OuterCryostatFiber': 'Far', 'Co60_OuterCryostatSupportResin': 'Far', 'Co60_OuterCryostatSupportFiber': 'Far', 'Co60_InnerCryostatResin': 'Far', 'Co60_InnerCryostatFiber': 'Far', 'Co60_InnerCryostatSupportResin': 'Far', 'Co60_
# InnerCryostatSupportFiber': 'Far', 'Co60_InnerCryostatLiner': 'Far', 'K40_OuterCryostatResin': 'Far', 'K40_OuterCryostatFiber': 'Far', 'K40_OuterCryostatSupportResin': 'Far', 'K40_OuterCryostatSupportFiber': 'Far', 'K40_InnerCryostatResin': 'Far',
# 'K40_InnerCryostatFiber': 'Far', 'K40_InnerCryostatSupportResin': 'Far', 'K40_InnerCryostatSupportFiber': 'Far', 'K40_InnerCryostatLiner': 'Far', 'U238_HFE': 'Vessel_U238', 'U238_TPCVessel': 'Vessel_U238', 'U238_TPCSupportCone': 'Vessel_U238', 'U23
# 8_HVTubes': 'Vessel_U238', 'U238_HVCables': 'Vessel_U238', 'U238_HVFeedthrough': 'Vessel_U238', 'U238_HVFeedthroughCore': 'Vessel_U238', 'U238_CalibrationGuideTube1': 'Vessel_U238', 'U238_CalibrationGuideTube2': 'Vessel_U238', 'Th232_HFE': 'Vessel_
# Th232', 'Th232_TPCVessel': 'Vessel_Th232', 'Th232_TPCSupportCone': 'Vessel_Th232', 'Th232_HVTubes': 'Vessel_Th232', 'Th232_HVCables': 'Vessel_Th232', 'Th232_HVFeedthrough': 'Vessel_Th232', 'Th232_HVFeedthroughCore': 'Vessel_Th232', 'Th232_Calibrati
# onGuideTube1': 'Vessel_Th232', 'Th232_CalibrationGuideTube2': 'Vessel_Th232', 'U238_Cathode': 'Internals_U238', 'U238_Bulge': 'Internals_U238', 'U238_FieldRings': 'Internals_U238', 'U238_SupportRodsandSpacers': 'Internals_U238', 'U238_SiPMStaves':
      config_file.close()



   ##########################################################################
   # Loads the input dataframe from the h5 file generated by the
   # ConvertExcel2DataFrame.py script
   ##########################################################################
   def LoadComponentsTableFromFile( self, input_filename ):

      print('\nLoading input data froma previously-generated components table....')

      if not (self.df_components.empty):
         print('\nWARNING: There is already an input dataframe loaded. It ' +\
               'will be overwritten.')

      try:
         self.df_components = pd.read_hdf(input_filename,key='Components')
      except OSError as e:
          print('\nERROR: The input file must be an HDF5 file.\n')
          #sys.exit()
          return
      except KeyError as e:
          print('\n')
          print(e)
          print('\nERROR: The input file should contain the component-wise activity and PDF ' +\
                'information. This can be generated from the Materials Database excel ' +\
                'spreadsheets using the ConvertExcel2DataFrame.py script.\n')
          #sys.exit()
          return

      print('\nLoaded dataframe with {} components.'.format(len(self.df_components)))
      print('Contains the following quantities of interest:')
      for column_name in self.df_components.columns:
          print('\t{}'.format(column_name))

      if 'HistogramAxisNames' in self.df_components.columns:     
         self.histogram_axis_names = self.df_components['HistogramAxisNames'].iloc[0]
         print('\nFit variables:\n' +\
               '\t{}'.format(self.histogram_axis_names))
      else:
         print('WARNING: We don\'t have axis names for the histograms. ' +\
               'Please ensure they are set in the config file, or we might run ' +\
               'into problems.')

      # print(f"df_components: {self.df_components}")
#       df_components:                              PDFName                       Component  Isotope   MC ID  Total Mass or Area  ...                                          Histogram                    HistogramAxisNames  TotalHitEff_K Group  Expected Co
# unts
# 0     U238_OuterCryostatSupportResin  Outer Cryostat Support (Resin)    U-238  MC-089               345.0  ...  Hist(2 bins in [0,2], 280 bins in [700.0,3500....  [SS/MS, Energy (keV), Standoff (mm)]   1.527000e+03   Far    -3.804523e+03
# 1    Th232_OuterCryostatSupportResin  Outer Cryostat Support (Resin)   Th-232  MC-089               345.0  ...  Hist(2 bins in [0,2], 280 bins in [700.0,3500....  [SS/MS, Energy (keV), Standoff (mm)]   4.512880e+03   Far    -2.636658e+04
# 2      K40_OuterCryostatSupportResin  Outer Cryostat Support (Resin)     K-40  MC-089               345.0  ...  Hist(2 bins in [0,2], 280 bins in [700.0,3500....  [SS/MS, Energy (keV), Standoff (mm)]   1.110000e+02   Far    -3.755857e+03
# 3     Co60_OuterCryostatSupportResin  Outer Cryostat Support (Resin)    Co-60  MC-089               345.0  ...  Hist(2 bins in [0,2], 280 bins in [700.0,3500....  [SS/MS, Energy (keV), Standoff (mm)]   7.850000e+02   Far    -3.202776e+03
# 4    Cs137_OuterCryostatSupportResin  Outer Cryostat Support (Resin)   Cs-137  MC-089               345.0  ...                                               None                                  None   0.000000e+00   Off    -1.000000e+00
# ..                               ...                             ...      ...     ...                 ...  ...                                                ...                                   ...            ...   ...              ...
# 147                 Th232_SolderSiPM                   Solder (SiPM)   Th-232  MC-082                 0.1  ...  Hist(2 bins in [0,2], 280 bins in [700.0,3500....  [SS/MS, Energy (keV), Standoff (mm)]   3.630256e+05   Off     2.003466e+06
# 148                   K40_SolderSiPM                   Solder (SiPM)     K-40  MC-082                 0.1  ...  Hist(2 bins in [0,2], 280 bins in [700.0,3500....  [SS/MS, Energy (keV), Standoff (mm)]   4.143400e+04   Off     4.873852e+05
# 149                  Co60_SolderSiPM                   Solder (SiPM)    Co-60  MC-082                 0.1  ...  Hist(2 bins in [0,2], 280 bins in [700.0,3500....  [SS/MS, Energy (keV), Standoff (mm)]   5.986080e+05   Off     3.020432e+05
# 150                Ag110m_SolderSiPM                   Solder (SiPM)  Ag-110m  MC-082                 0.1  ...  Hist(2 bins in [0,2], 280 bins in [700.0,3500....  [SS/MS, Energy (keV), Standoff (mm)]   3.416731e+07   Off     1.939501e+06
# 151                 Cs137_SolderSiPM                   Solder (SiPM)   Cs-137  MC-082                 0.1  ...                                               None                                  None   0.000000e+00   Off    -1.000000e+00
#
# [152 rows x 17 columns]

      return
   ################# End of LoadComponentsTableFromFile() ###################



   ##########################################################################
   # Decide how to handle the radioassay measurements. By default, the 
   # central values will be used (if available) or the 90% upper limits,
   # with no fluctuations. There are two arguments:
   #      -  'fluctuate' allows the code to randomly sample a radioassay
   #          result from a gaussian distribution centered at the central 
   #          value, but it will truncate the distribution at 0 (so we 
   #          don't have negative counts).
   #      -  'limits_handling' governs how we deal with radioassay results
   #          that only report an upper limit. The default is "Delta", where
   #          we simply use the upper limit as the result, and apply no
   #          fluctuations. The other option is "TruncatedGaussianAtZero",
   #          where we assume the central value is 0, then sample from a 
   #          gaussian that has a sigma given by assuming the reported value
   #          is the 90% CL. 
   ##########################################################################
   def SetHandlingOfRadioassayData( self, fluctuate = False, limits_handling = "Delta" ):
       self.fluctuate_radioassay_during_grouping = fluctuate
       self.radioassay_limits_handling = limits_handling 


   ##########################################################################
   # Creates the components table from the Excel spreadsheet.
   ##########################################################################
   def CreateComponentsTableFromXLS( self, excelFile, histogramsFile ):

      # Check and see if we're overwriting something here.
      if not (self.df_components.empty):
         print('\nWARNING: There is already an input dataframe loaded. It ' +\
               'will be overwritten.')
 
      start_time = time.time()      

      excelTableReader = nEXOExcelTableReader.nEXOExcelTableReader(excelFile, \
                                                                   histogramsFile, \
                                                                   config = self.config)
      try: 
         excelTableReader.ConvertExcel2DataFrame()
      except KeyError:
         sys.exit()
   
      self.df_components = excelTableReader.components

      # Print out some useful info.
      print('\nLoaded dataframe with {} components.'.format(len(self.df_components)))
      print('Contains the following quantities of interest:')
      for column_name in self.df_components.columns:
          print('\t{}'.format(column_name))
      if 'HistogramAxisNames' in self.df_components.columns:     
         self.histogram_axis_names = self.df_components['HistogramAxisNames'].iloc[0]
         print('\nFit variables:\n' +\
               '\t{}'.format(self.histogram_axis_names))
      else:
         print('WARNING: We don\'t have axis names for the histograms. ' +\
               'Please ensure they are set in the config file, or we might run ' +\
               'into problems.')

      # Store the components table in a file in case you want to
      # go back and look at it later.
      nameTag = excelTableReader.GetExcelNameTag( excelFile )
      outTableName = 'ComponentsTable_' + nameTag + '.h5'
      print('\n\nWriting table to file {}\n'.format(outTableName))
      excelTableReader.components.to_hdf( outTableName, key='Components' )

      end_time = time.time()
      print('Elapsed time = {:3.3} seconds ({:3.3} minutes).'.format( \
                              end_time-start_time, \
                             (end_time-start_time)/60. ) )
      return


   ##########################################################################
   # Creates the components by connecting directly to the Materials DB.
   # Geometry tag is the number associated with a particular model, e.g.
   # 'D-023' for the Baseline 2019 geometry
   ##########################################################################
   def CreateComponentsTableFromMaterialsDB( self, geometry_tag, label='',\
                                             histograms_file = None,\
                                             create_histograms_from_trees = False,\
                                             trees_directory = None,\
                                             download_mc_data = False,\
                                             use_mc_id_dict = False):

       start_time = time.time()

       if use_mc_id_dict:
          mc_id_dict = MC_ID_Dict()

       if histograms_file is None and\
          not create_histograms_from_trees and\
          not download_mc_data:
          print('\nNo histogram data can be found! Please do one of the following:')
          print('\t- Specify a hisograms_file')
          print('\t- Set create_histograms_from_trees = True and specify a trees_directory')
          print('\t- Set download_mc_data = True (trees will be downloaded from the DB)')
          print('\nAborting without creating a ComponentsTable...\n')
          return
       elif not create_histograms_from_trees and not download_mc_data:
          df_mc_histograms = pd.read_hdf( histograms_file, key='SimulationHistograms' )
       if create_histograms_from_trees and trees_directory is None:
          print('\ncreate_histograms_from_trees is True, but no trees_directory is specified')
          print('Please specify the path to the data trees')
          print('\nAborting without creating a ComponentsTable...\n')
          return

       if download_mc_data:
          print('\nDirect download of MC root files is not yet implemented.')
          print('Please locate the root files manually and try again')
          print('\nAborting without creating a ComponentsTable...\n')
          return



       
       if self.materialsDB is None:
          self.ConnectToMaterialsDB()
       
       self.df_components = pd.DataFrame()

       geometry_doc = self.materialsDB.GetTaggedGeometry( geometry_tag )
       print('\nRetrieved data for geometry {}: {}'.format(geometry_tag,geometry_doc['title']))

       counter = 0

       for component in geometry_doc['components']:
           print('\n{}'.format(component['name']))
           print('    - Material: {}'.format(component['material']))
           print('    - Mass/Area: {} {} x {} units'.format(component['mass'],component['unit'],component['quantity']))
           print('    - Isotopes and activities:')

           radioassay_data = self.materialsDB.GetRadioassayData( component['radioassayid'] )
           data_type = component['radioassayid'][0]           

           for measurement in radioassay_data:

               thispdf = pd.Series()

               if data_type == 'R':
                  print('          * {:<17} {:>10.9} +/- {:<10.9} {:<10} ({})'.format( \
                                                                       measurement['isotope']+':', \
                                                                       measurement['specific_activity'],\
                                                                       measurement['error'],\
                                                                       measurement['unit'],\
                                                                       measurement['error_type']) )
               elif data_type == 'P':
                  print('          * {:<17} {:>10.9} +/- {:<10.9} {:<10} ({})'.format( \
                                                                       measurement['isotope']+':', \
                                                                       measurement['value'],\
                                                                       measurement['error'],\
                                                                       measurement['unit'],\
                                                                       measurement['type']) )
               else:
                  print('Unknown Radioassay ID: {}'.format(component['radioassayid']))
                  print('It will not be included in the analysis.')
                  continue
                 
               # Remove all spaces, parentheses, and hyphens from the names
               component_name = component['name'].replace('(','').replace(')','').replace(' ','')
               isotope_name = ( measurement['isotope'].split(' ')[0] ).replace('-','')

               pdf_name = '{}_{}'.format(isotope_name,component_name)
               print(pdf_name) 
               thispdf['PDFName'] = pdf_name
               thispdf['Component'] = component['name']
               thispdf['Isotope'] = measurement['isotope'].split(' ')[0]
               if use_mc_id_dict:
                  thispdf['MC ID'] = mc_id_dict.data[component['name']]
               else:
                  thispdf['MC ID'] = component['montecarloid']
               print(thispdf['MC ID'],'-----MC-ID---fys')
               thispdf['Total Mass or Area'] = float( component['mass'] ) * float( component['quantity'] )
               thispdf['Activity ID'] = component['radioassayid'] 

               if data_type == 'R':
                  thispdf['SpecActiv'] = self.materialsDB.ConvertSpecActivTo_mBqPerUnitSize( measurement['specific_activity'],\
                                                                                             measurement['unit'],\
                                                                                             measurement['isotope'] )
                  thispdf['SpecActivErr'] = self.materialsDB.ConvertSpecActivTo_mBqPerUnitSize( measurement['error'],\
                                                                                                measurement['unit'],\
                                                                                                measurement['isotope'] )
                  thispdf['SpecActivErrorType'] = measurement['error_type'] # either a 90%CL or a gaussian error bar

               elif data_type == 'P':
                  thispdf['SpecActiv'] = self.materialsDB.ConvertSpecActivTo_mBqPerUnitSize( measurement['value'],\
                                                                                             measurement['unit'],\
                                                                                             measurement['isotope'] )
                  thispdf['SpecActivErr'] = self.materialsDB.ConvertSpecActivTo_mBqPerUnitSize( measurement['error'],\
                                                                                                measurement['unit'],\
                                                                                                measurement['isotope'] )
                  thispdf['SpecActivErrorType'] = measurement['type'] # either a 'limit' or an observation ('obs')

               thispdf['RawActiv'] = thispdf['SpecActiv'] * thispdf['Total Mass or Area']
               thispdf['RawActivErr'] = thispdf['SpecActivErr'] * thispdf['Total Mass or Area']
      
               # Get the number of primaries from the Materials DB
               mc_data = self.materialsDB.GetDoc( thispdf['MC ID'] )
               thispdf['TotalHitEff_N'] = 0.
               try:
                    found = 0
                    for isotope_mc in mc_data['rootfiles']:
                        if thispdf['Isotope'] in isotope_mc['isotope']:
                           thispdf['TotalHitEff_N'] = float(isotope_mc['numbersimed'])
                           found += 1
                    if found == 0:
                        print('\n\tWARNING: No match for {} inlist of rootfiles for {}'.format(thispdf['Isotope'],thispdf['MC ID']))
               except KeyError:
                    print('\nWARNING: no root files for MC ID: {}\n'.format(thispdf['MC ID']))

               # Get the histogram and histogram axis names
               thispdf['Histogram'] = None
               thispdf['HistogramAxisNames'] = None

               for index, row in df_mc_histograms.iterrows():

                   #print(isotope_name,'----fys------isotope name---')
                   if isotope_name in row['Filename'] and\
                      thispdf['MC ID'] in row['Filename']:
                        print(row['Filename'],'----fys------filename---')
                        thispdf['Histogram'] = row['Histogram']
                        print(thispdf['Histogram'],'----fys-------histogram---')
                        thispdf['HistogramAxisNames'] = row['HistogramAxisNames']
               if thispdf['Histogram'] is None:
                   print('\n          WARNING: NO MC HISTOGRAM FOUND FOR {}'.format( thispdf['PDFName'] ))
                   print('          Isotope: {}\t MC ID: {}\n'.format(isotope_name, thispdf['MC ID'] ))
             
               # Get tie integral in the input histogram
               if thispdf['Histogram'] is None:
                   thispdf['TotalHitEff_K'] = 0.
               else:
                   thispdf['TotalHitEff_K'] = np.sum( thispdf['Histogram'].values ) 

               try:
                   thispdf['Group'] = self.config['GroupAssignments'][ thispdf['PDFName'] ]
               except KeyError:
                   print('\n\t*************************** ERROR ***********************************\n' + \
                         '\tThere is no group assignment for {}\n'.format(thispdf['PDFName']) + \
                         '\tPlease add one to the configuration file and try again.\n')
                   raise KeyError

               if thispdf['TotalHitEff_N'] > 0.:
                   thispdf['Expected Counts'] =  thispdf['TotalHitEff_K'] / thispdf['TotalHitEff_N'] * \
                                                 thispdf['RawActiv'] * 60. * 60. * 24. * 365. * 10
               else:
                   thispdf['Expected Counts'] = -1.

               if self.df_components.empty:
                  self.df_components = pd.DataFrame(columns=thispdf.index.values)
               self.df_components.loc[counter] = thispdf
               counter += 1
  
       # Print out some useful info.
       print('\nLoaded dataframe with {} components.'.format(len(self.df_components)))
       print('Contains the following quantities of interest:')
       for column_name in self.df_components.columns:
           print('\t{}'.format(column_name))
       if 'HistogramAxisNames' in self.df_components.columns:     
          self.histogram_axis_names = self.df_components['HistogramAxisNames'].iloc[0]
          print('\nFit variables:\n' +\
                '\t{}'.format(self.histogram_axis_names))
       else:
          print('WARNING: We don\'t have axis names for the histograms. ' +\
                'Please ensure they are set in the config file, or we might run ' +\
                'into problems.')
 
       # Store the components table in a file in case you want to
       # go back and look at it later.
       outTableName = 'ComponentsTable_' + geometry_tag + '_{}'.format(label) + '.h5'
       print('\n\nWriting table to file {}\n'.format(outTableName))
       self.df_components.to_hdf( outTableName, key='Components' )
 
       end_time = time.time()
       print('Elapsed time = {:3.3} seconds ({:3.3} minutes).'.format( \
                               end_time-start_time, \
                              (end_time-start_time)/60. ) )
       return


   ##########################################################################
   # Creates a nEXOMateiralsDBInterface object, which connects to the DB
   # upon construction
   ##########################################################################
   def ConnectToMaterialsDB( self ):
       self.materialsDB = nEXOMaterialsDBInterface.nEXOMaterialsDBInterface()
       self.materialsDB.PrepareDB()



   ##########################################################################
   # Creates the binned PDFs from the TTrees produced by the Reconstruction.
   # Stores the histograms as histlite objects, then writes these to an HDF5
   # file.
   ##########################################################################
   def CreateHistogramsFromRawTrees( self, path_to_trees, output_hdf5_filename ):

      start_time = time.time()      

      num_rootfiles = len(os.listdir(path_to_trees)) 

      rows_list = []
      num_processed = 0
      
      for filename in os.listdir(path_to_trees):
          num_processed += 1
          if '.root' not in filename:
             continue
          print('Loading {} at {:.4} seconds...\t({}/{})'.format(filename,\
                                                        time.time()-start_time,\
                                                        num_processed,\
                                                        num_rootfiles))
          thisfile = uproot.open( (path_to_trees + '/' + filename) )
          #print(thisfile.keys())
          #print(thisfile["Event"].keys())
          #print(thisfile["Event/Recon"].values())
          #print(thisfile["Event/Recon/Standoff/Standoff/lower_z"].values())
          #print("Dubug")
          try:
             input_tree = thisfile['tree']
          except KeyError:
             print('\n\n************ ERROR: PROBLEM WITH FILE FORMAT ****************')
             print('The TTree in the file is not named \'tree\'.')
             print('Double-check that you\'re using the files with the reduced,')
             print('flattened trees.')
             print('\n*********************** EXITING *******************************\n')
             sys.exit('')

          # Grab the correct variables from the tree, and also grab the 'weight' column
          #var_list = [axis['VariableName'] for axis in self.config['FitAxes']].append('weight')
          var_list = self.GetReconVariableList()         

          try:
             input_df = input_tree.arrays( var_list, outputtype=pd.DataFrame )
          except KeyError as e:
             print('\n\n************** ERROR: PROBLEM WITH AXIS NAMES **************')
             print('The input TTree does not contain all of the variables listed')
             print('in the FitAxes. See error message below for details:\n')
             print(e)
             print('\n*********************** EXITING *******************************\n')
             sys.exit('') 

          # Define binning
          binspecs_list = []
          for axis in self.config['FitAxes']:
              if 'BinningOption' not in axis.keys():
                  print('\n\n************** ERROR: PLEASE SEPCIFY A BINNING OPTION **************')
                  print('Supported options are:')
                  print('      \'Linear\': requires a min, max, and number of bins')
                  print('      \'Custom\': requires a list of the bin edges')
                  print('\n*********************** EXITING *******************************\n')
                  sys.exit('') 

              if 'Linear' in axis['BinningOption']:
                  binspecs_list.append( np.linspace( axis['Min'],\
                                                    axis['Max'],\
                                                    axis['NumBins']+1 ) )
              elif 'Custom' in axis['BinningOption']:
                  binspecs_list.append( np.array(axis['BinEdges']) )
              else:   
                  print('\n\n************** ERROR: {} IS NOT A SUPPORTED BINNING OPTION **************'.format(axis['BinningOption']))
                  print('Supported options are:')
                  print('      \'Linear\': requires a min, max, and number of bins')
                  print('      \'Custom\': requires a list of the bin edges')
                  print('\n*********************** EXITING *******************************\n')
                  sys.exit('') 


          # Define cuts
          mask = self.GetCutMask( input_df )
          if len(mask) > 0:
             print('\tEvents passing cuts: {} evts of {} ({:4.4}%)'.format(\
                     np.sum(mask), len(mask), 100*float(np.sum(mask))/float(len(mask))))
 
          data_list = [ input_df[ axis['VariableName'] ].loc[mask] for axis in self.config['FitAxes'] ]
          hh = hl.hist( tuple(data_list), weights=input_df['weight'].loc[mask], bins=tuple(binspecs_list) )
          thisrow = { 'Filename': filename,\
                      'Histogram': hh,\
                      'HistogramAxisNames': [ axis['Title'] for axis in self.config['FitAxes'] ] }
          rows_list.append(thisrow)

      df_pdf = pd.DataFrame(rows_list)
      df_pdf.to_hdf( output_hdf5_filename, key='SimulationHistograms' )

      end_time = time.time()
      print('Elapsed time = {} seconds ({} minutes).'.format( end_time-start_time, \
                                                              (end_time-start_time)/60. ))
   


   ##########################################################################
   # Returns a list containing the names of all the variables you need to
   # grab from the reconstructed data.
   ##########################################################################
   def GetReconVariableList( self ):
      recon_variable_list = []
      
      # Add whatever variables will go on the axes of your PDFs
      for axis in self.config['FitAxes']:
          recon_variable_list.append( axis['VariableName'] )

      # Add whatever variables you will need to cut on
      for cut in self.config['Cuts']:
          for propertyname, value in cut.items():
              if 'Variable' in propertyname:
                 recon_variable_list.append( value )

      # Make sure you also grab the weights
      recon_variable_list.append( 'weight' )
    
      # Remove any duplicate variable names
      recon_variable_list = list( dict.fromkeys( recon_variable_list ) )

      return recon_variable_list
     

   ##########################################################################
   # Generates a boolean numpy array that represents the sum of all the cuts
   # to apply to the reconstructed data while making the PDFs.
   ##########################################################################
   def GetCutMask( self, input_df ):

      global_mask = np.ones( len(input_df), dtype=bool)

      for cut in self.config['Cuts']:

          if cut['Type'] == 'Boolean':
             this_mask = input_df[ cut['Variable'] ].values

          elif cut['Type'] == 'InvertedBoolean':
             this_mask = np.invert( input_df[ cut['Variable'] ].values )

          elif cut['Type'] == '1D':
             if cut['Bound'] == 'Upper':
                this_mask = input_df[cut['Variable']].values < cut['Value']
             elif cut['Bound'] == 'Lower':
                this_mask = input_df[cut['Variable']].values > cut['Value']
             else:
                print('\n\n************** ERROR: UNSUPPORTED CUT BOUND **************')
                print('\'Bound\' type {} is not supported. '.format(cut['Bound']) +\
                      'Please choose either \'Upper\' or \'Lower\'.')
                print('\n*********************** EXITING *******************************\n')
                sys.exit('') 
                

          elif cut['Type'] == '2DLinear':
             if cut['Bound'] == 'Upper':
                this_mask = input_df[cut['YVariable']].values < \
                              cut['Slope'] * input_df[cut['XVariable']].values + cut['Intercept']
             elif cut['Bound'] == 'Lower':
                this_mask = input_df[cut['YVariable']].values > \
                              cut['Slope'] * input_df[cut['XVariable']].values + cut['Intercept']
             else:
                print('\n\n************** ERROR: UNSUPPORTED CUT BOUND **************')
                print('\'Bound\' type {} is not supported. '.format(cut['Bound']) +\
                      'Please choose either \'Upper\' or \'Lower\'.')
                print('\n*********************** EXITING *******************************\n')
                sys.exit('')
 
          else:
             print('\n\n************** ERROR: UNSUPPORTED CUT **************')
             print('Cut type {} is not yet supported. ' +\
                   'See nEXOFitWorkspace.GetCutMask for more details.'.format(cut['Type']))
             print('\n*********************** EXITING *******************************\n')
             sys.exit('') 

          global_mask = global_mask & this_mask

      return global_mask  
          
 
   ##########################################################################
   # Creates grouped PDFs from the information contained in the input
   # dataframe.
   ##########################################################################
   def CreateGroupedPDFs( self ):

       print('\nCreating grouped PDFs....')
      
       if not (self.df_group_pdfs.empty):
          print('\nWARNING: Group PDFs have already been generated. ' +\
                'They are going to be overwritten.')
 
       self.df_group_pdfs = pd.DataFrame(columns = ['Group',\
                                                    'Histogram',\
                                                    'TotalExpectedCounts'])

       # Loop over rows in df_components, add histograms to the appropriate group.
       for index,row in self.df_components.iterrows():

         if row['Group'] == 'Off':
            continue

         # If the histogram is non-zero, normalize it. Note that running
         # `.normalize` on a histogram with all zeros returns a histogram filled with nan
         print(row['PDFName'],'---fys----PDFName')
         if np.sum( row['Histogram'].values ) > 0.:
            #print(row['Histogram'].values,'---fys----histogram---values')
            print(row['Histogram'],'---fys----histogram---')
            normalized_histogram = row['Histogram'].normalize( (0,1,2), integrate=False ) 
         else:
            normalized_histogram = row['Histogram']

         if row['Isotope']=='bb0n':
             totalExpectedCounts = self.signal_counts
         elif self.fluctuate_radioassay_during_grouping:
             totalExpectedCounts = self.GetFluctuatedExpectedCounts( row ) 
         else:
             if row['SpecActiv'] > 0.:
                totalExpectedCounts = row['Total Mass or Area'] * \
                                      row['SpecActiv']/1000. * \
                                      row['TotalHitEff_K'] / row['TotalHitEff_N'] * \
                                      self.livetime
             else:
                totalExpectedCounts = 0.
         
         
         if not (row['Group'] in self.df_group_pdfs['Group'].values):

             new_group_row = { 'Group' : row['Group'], \
                               'Histogram' : normalized_histogram * \
                                             totalExpectedCounts, \
                               'TotalExpectedCounts' : totalExpectedCounts }

             self.df_group_pdfs = self.df_group_pdfs.append(new_group_row,ignore_index=True)
 
         else:

             group_mask = row['Group']==self.df_group_pdfs['Group']            
 
             self.df_group_pdfs.loc[ group_mask, 'Histogram' ] = \
                  self.df_group_pdfs['Histogram'].loc[ group_mask ] + \
                  normalized_histogram * \
                  totalExpectedCounts

             self.df_group_pdfs.loc[ group_mask,'TotalExpectedCounts'] = \
                  self.df_group_pdfs['TotalExpectedCounts'].loc[ group_mask ] + \
                  totalExpectedCounts

       # One more loop accomplishes two things:
       #     1. Normalize the grouped PDFs (now that each component has been added with
       #        the correct weight)
       #     2. Generate a summed PDF (including the expected weights) and append it.
       total_sum_row = {}
       for index,row in self.df_group_pdfs.iterrows():

         if np.sum( row['Histogram'].values ) > 0.:
            self.df_group_pdfs.loc[ index, 'Histogram'] = row['Histogram'].normalize( (0,1,2), integrate=False )

         # TOFIX: Need better handling of negative totals, and need to figure out why 'Far' gives
         # me weird stuff. For now, I'm ignoring these.
         if (row['TotalExpectedCounts']>0.)&(row['Group']!='Off'): #&(row['Group']!='Far'):
             if len(total_sum_row)==0:
                 total_sum_row = {'Group' : 'Total Sum',\
                                  'Histogram' : row['Histogram'],\
                                  'TotalExpectedCounts' : row['TotalExpectedCounts']}
             else:
                 total_sum_row['Histogram'] = total_sum_row['Histogram'] + row['Histogram']
                 total_sum_row['TotalExpectedCounts'] = total_sum_row['TotalExpectedCounts'] + row['TotalExpectedCounts']
      
       self.df_group_pdfs = self.df_group_pdfs.append(total_sum_row,ignore_index=True)

       # Print out all the groups and their expected counts.
       print('\t{:<20} \t{:>15}'.format('Group:','Expected Counts:'))
       for index,row in self.df_group_pdfs.iterrows():
           print('\t{:<20} \t{:>15.2f}'.format( row['Group'], row['TotalExpectedCounts']))
#        print(f"df_group_pdfs: {self.df_group_pdfs}")
#        df_group_pdfs:               Group                                          Histogram  TotalExpectedCounts
# 0               Far  Hist(2 bins in [0,2], 280 bins in [700.0,3500....         4.920165e+03
# 1      Vessel_Th232  Hist(2 bins in [0,2], 280 bins in [700.0,3500....         2.977887e+03
# 2       Vessel_U238  Hist(2 bins in [0,2], 280 bins in [700.0,3500....         2.185273e+04
# 3    Internals_U238  Hist(2 bins in [0,2], 280 bins in [700.0,3500....         4.635314e+04
# 4   Internals_Th232  Hist(2 bins in [0,2], 280 bins in [700.0,3500....         1.021549e+04
# 5      FullTPC_Co60  Hist(2 bins in [0,2], 280 bins in [700.0,3500....         2.161884e+02
# 6       FullTPC_K40  Hist(2 bins in [0,2], 280 bins in [700.0,3500....         3.249165e+07
# 7             Rn222  Hist(2 bins in [0,2], 280 bins in [700.0,3500....         9.107267e+03
# 8       FullLXeBb2n  Hist(2 bins in [0,2], 280 bins in [700.0,3500....         2.794938e+07
# 9       FullLXeBb0n  Hist(2 bins in [0,2], 280 bins in [700.0,3500....         1.000000e-04
# 10            Xe137  Hist(2 bins in [0,2], 280 bins in [700.0,3500....         4.651665e+01
# 11        Total Sum  Hist(2 bins in [0,2], 280 bins in [700.0,3500....         6.053672e+07

       return 
   ######################## End of CreateGroupPDFs() ########################


   ##########################################################################
   # Apply fluctuation

   ##########################################################################
   def GetFluctuatedExpectedCounts( self, components_table_row ):
       
       # Fluctuations are applied in two ways:
       #    - If the radioassay result is a 90% upper limit, we draw from a 
       #      gaussian centered at 0, with 1-sigma = value/sqrt(2)/erfinv(0.8)
       #    - If the radioassay result is a measured value, we draw from a gaussian
       #      centered at the measured value with the measured 1-sigma uncertainties. 
       #    - If the fluctuated value is below 0, resample until you get a nonnegative result. 
       # This strategy follows the recommendations in Raymond's paper: arXiv:1808.05307

       fluctuated_spec_activity = None

       if components_table_row['SpecActivErrorType'] == 'Upper limit (90% C.L.)' or \
          components_table_row['SpecActivErrorType'] == 'limit':
           fluct_mean = 0.
           fluct_sigma = components_table_row['SpecActiv'] / np.sqrt(2) / 0.906194
       elif components_table_row['SpecActivErrorType'] == 'Symmetric error (68% C.L.)' or \
            components_table_row['SpecActivErrorType'] == 'obs':
               fluct_mean = components_table_row['SpecActiv']
               fluct_sigma = components_table_row['SpecActivErr']
       else: 
           print('\nComponent {} has undefined error type {}\n'.format(\
                            components_table_row['SpecActivErrorType'],\
                            components_table_row['SpecActivErrorType'] ))
           raise TypeError
   
       # keep trying until you get a nonnegative number
       fluctuated_spec_activity = np.random.normal(fluct_mean,fluct_sigma)
       if fluctuated_spec_activity < 0.:
              fluctuated_spec_activity = 0.
              #fluctuated_spec_activity = np.random.normal(fluct_mean,fluct_sigma)

       totalExpectedCounts = components_table_row['Total Mass or Area'] * \
                             fluctuated_spec_activity/1000. * \
                             components_table_row['TotalHitEff_K'] / \
                             components_table_row['TotalHitEff_N'] * \
                             self.livetime

       return totalExpectedCounts
        

   ##########################################################################
   # Defines the ROI. The ROI boundaries are given as a dict, where the
   # keys are the axis names and the values are a tuple or list defining
   # the boundary values. For example, if my fit axes are SS/MS, Energy, 
   # and standoff, I could give it:
   #      input_dict = { 'SS/MS':  [ 0., 1.] ,
   #                     'Energy': [ 2428.89, 2486.77 ] ,
   #                     'Standoff' [ 120., 650. ] }
   ##########################################################################
   def DefineROI( self, input_roi_dict ):
      
       # First, make sure the right axes are being called.
       for axis in input_roi_dict.keys():
           if not axis in self.histogram_axis_names:
              print('ERROR: {} does not match any of the fit axes.')
              print('       Choices are:')
              for name in self.histogram_axis_names:
                  print('                 {}'.format(name))
              print('Failed to set the ROI.')
              return
         
       # Next, find the bin edges that are closest to these points.
       pdf_bins = self.df_group_pdfs['Histogram'].iloc[0].bins

       self.roi_indices = dict()
       self.roi_edges = dict()
       
       for i in range(len(pdf_bins)):

           axis_name = self.histogram_axis_names[i]
           axis_bins = pdf_bins[i]

           match_edges_lower_limit = np.where( axis_bins >= input_roi_dict[axis_name][0] )
           match_edges_upper_limit = np.where( axis_bins <= input_roi_dict[axis_name][1] )
   
           match_edges = np.intersect1d( match_edges_lower_limit, match_edges_upper_limit )
           match_indices = match_edges[:-1]
 
           self.roi_edges[axis_name] = np.array( axis_bins[match_edges] )
           self.roi_indices[axis_name] = np.array( match_indices )
           print('{}:'.format(axis_name))
           print('\tInput ROI boundaries:  {:>8.5}, {:>8.5}'.format(\
                  float(input_roi_dict[axis_name][0]), float(input_roi_dict[axis_name][1]) ) )
           print('\tActual ROI boundaries: {:>8.5}, {:>8.5}'.format(\
                  float(self.roi_edges[axis_name][0]), float(self.roi_edges[axis_name][-1]) ) )


   #################### End of DefineROI() ##################################        


   
   ##########################################################################
   # Print the edges of the ROI, as defined by the binning of the PDFs.
   # Note that this will not be exactly the same as the input ROI.
   ##########################################################################
   def PrintROIEdges( self ):

       print('\n********************************************************')
       print(' ROI bin edges:')       

       for axis_name, axis_edges in self.roi_edges.items():
           print('\t{:<15} {:>8.5}, {:>8.5}'.format( axis_name+':',\
                 float(axis_edges[0]), float(axis_edges[-1]) ) )
       print('********************************************************')

       print('\n')
   ##################### End of PrintROIEdges() ############################



   #########################################################################
   # Return the indices, in the correct array structure, of the bins
   # in the ROI
   #########################################################################
   def GetROIBinIndices( self ):

       roi_indices_array = []

       for i in range( len(self.histogram_axis_names) ):
           axis_name = self.histogram_axis_names[i]
           axis_indices = self.roi_indices[axis_name]
           roi_indices_array.append( axis_indices )

       return np.array(roi_indices_array)
   ###################### End of GetROIBinIndices() #######################







   ##########################################################################
   # Creates negative log likelihood object
   ##########################################################################
   #def CreateNegLogLikelihood( self ):
       



