########################################################################
##
## Class for reading the spreadsheets produced by the materials database.
##
########################################################################


import pandas as pd
import numpy as np
import NameDict
import yaml
#import openpyxl

######################################################
############### EXCEL TABLE READER ###################
######################################################
class nEXOExcelTableReader:
    # This class carries all table info
    # Modify '__init__' appropriately for table in use  

    def __init__(self,inTableName,pathToPDFs,config='./config/TUTORIAL_config.yaml'):

        self.DEBUG=False

        self.filename = inTableName

        self.df_pdfs = pd.read_hdf( pathToPDFs, key='SimulationHistograms' )

        self.quantile = 1.64

        #self.suffixes = ['SS','MS']

        self.specActivSheet = 'SpecificActivities'
        self.detectorSpecsSheet = 'DetectorSpecifications'
        self.countsSheet = '%s_ExpectedCounts'
        self.hitEffSheet = 'MC_RawCounts_%s_Integrals'
        self.halflifeSheet = 'Halflives'                

        # Load in the configuration data, which contains the group assignments
        if isinstance( config, str ):
           config_file = open( config, 'r' )
           self.config = yaml.load( config_file, Loader=yaml.SafeLoader )
           config_file.close()
        else:
           self.config = config

        
        # END OF CONSTRUCTOR

    ##########################################################################
    # Reads the necessary data from the .xlsx spreadsheet file into a single
    # pandas dataframe. This replaces the old C++ class.
    ##########################################################################
    def ConvertExcel2DataFrame( self ):

        print('Loading sheets...')
        self.dfSpecActiv = self.ReadBasicSheet( self.specActivSheet )
        self.dfDetSpecs = self.ReadBasicSheet( self.detectorSpecsSheet )
        #self.dfHalflife = self.ReadBasicSheet( self.halflifeSheet )
        self.dfHalflife = pd.read_excel( self.filename, sheet_name=self.halflifeSheet, header=None )
        self.dfHalflife.columns = ['Isotopes','Halflife (yrs)']
        self.dfCountsSS = self.ReadNestedSheet( self.countsSheet % 'SS' )
        self.dfCountsMS = self.ReadNestedSheet( self.countsSheet % 'MS' )
        self.dfHitEffSS = self.ReadNestedSheet( self.hitEffSheet % 'SS' )
        self.dfHitEffMS = self.ReadNestedSheet( self.hitEffSheet % 'MS' )
        print('Sheets loaded.')

        self.name_dict = {y:x for x,y in NameDict.NameDict().data.items()}

        self.components = pd.DataFrame()
        for index, row in self.dfSpecActiv.iterrows():
            thispdf = pd.Series()

            component = row['Component']
            isotope   = row['Isotope']

            # Remove all spaces, parentheses, and hyphens from the names for better bookkeeping.
            component_name = component.replace('(','').replace(')','').replace(' ','')
            isotope_name = ( isotope.split(' ')[0] ).replace('-','')
            

            pdf_name = '{}_{}'.format(isotope_name,component_name)

            thispdf['PDFName'] = pdf_name
            thispdf['Component'] = row['Component']
            thispdf['Isotope'] = row['Isotope']
            thispdf['MC ID'] = row['Monte Carlo']
            if self.DEBUG:
               print('Component: %s\t Isotope: %s\t MC_ID: %s' % (thispdf['Component'], thispdf['Isotope'], thispdf['MC ID']))

            thispdf['Histogram'] = self.df_pdfs['Histogram'].loc[ \
                                     (self.df_pdfs['Filename'].str.contains(thispdf['MC ID'])) & \
                                     (self.df_pdfs['Filename'].str.contains(thispdf['Isotope'].replace('-',''))) \
                                                          ].values[0]
            thispdf['HistogramAxisNames'] = self.df_pdfs['HistogramAxisNames'].loc[ \
                                     (self.df_pdfs['Filename'].str.contains(thispdf['MC ID'])) & \
                                     (self.df_pdfs['Filename'].str.contains(thispdf['Isotope'].replace('-',''))) \
                                                          ].values[0]

            # Set total mass of this component in the detector
            thispdf['Total Mass or Area'] = self.dfDetSpecs['Total Mass or Area'].loc[ self.dfDetSpecs['Component']==component ].values[0]

            # Set halflives
            df = self.dfHalflife
            thispdf['Halflife'] = df.loc[ df['Isotopes']==isotope, 'Halflife (yrs)'].iloc[0]

            # Setting activities.
            df = self.dfSpecActiv
            thisrow = ( df.loc[ df['Component']==component ] ).loc[ df['Isotope']==isotope ]
            thispdf['SpecActiv']    = thisrow[ 'Specific Activity [mBq/kg]' ].iloc[0]
            thispdf['SpecActivErr'] = thisrow[ 'Error [mBq/kg]' ].iloc[0]
            thispdf['RawActiv']     = thisrow[ 'Activity [Bq]' ].iloc[0]
            thispdf['RawActivErr']  = thisrow[ 'Error [Bq]' ].iloc[0]
            thispdf['Activity ID']     = thisrow[ 'Source' ].iloc[0]

            # Setting SS counts
            df = self.dfCountsSS['>700 keV (700, 3500)']['3.648']
            thisrowSS = (df.loc[ df['Component']==component ]).loc[ df['Isotope']==isotope ]
            df = self.dfCountsMS['>700 keV (700, 3500)']['3.648']
            thisrowMS = (df.loc[ df['Component']==component ]).loc[ df['Isotope']==isotope ]

            thispdf['Expected Counts'] = thisrowSS['C.V.'].iloc[0] + thisrowMS['C.V.'].iloc[0]
            thispdf['Expected Counts Err'] = np.sqrt( thispdf['Expected Counts'] ) #thisrowSS['Error'].iloc[0] + thisrowMS['Error'].iloc[0]
            thispdf['Expected Counts UL'] = thisrowSS['Upper Limit'].iloc[0] + thisrowMS['Upper Limit'].iloc[0]

           
            # Setting SS hit efficiencies
            # First, set the n and k variables
            df = self.dfHitEffSS['>700 keV (700, 3500)']['3.648']
            thisrowSS = (df.loc[ df['Component']==component ]).loc[ df['Isotope']==isotope ]
            df = self.dfHitEffMS['>700 keV (700, 3500)']['3.648']
            thisrowMS = (df.loc[ df['Component']==component ]).loc[ df['Isotope']==isotope ]
            thispdf['TotalHitEff_N'] = thisrowSS['No. of Disint.'].iloc[0]
            thispdf['TotalHitEff_K'] = thisrowSS['C.V.'].iloc[0] + thisrowMS['C.V.'].iloc[0]
            # Set the group name.
            try:
                   thispdf['Group'] = self.config['GroupAssignments'][ thispdf['PDFName'] ]
            except KeyError:
                   print('\n\t*************************** ERROR ***********************************\n' + \
                         '\tThere is no group assignment for {}\n'.format(thispdf['PDFName']) + \
                         '\tPlease add one to the configuration file and try again.\n')
                   raise KeyError
                    

            if self.DEBUG:
              print('Halflife: {}\tSpecActiv: {}\tSpecActivErr: {}\tRawActiv: {}\tAct.ID: {}'.format( \
                    thispdf['Halflife'],\
                    thispdf['SpecActiv'],\
                    thispdf['SpecActivErr'],\
                    thispdf['RawActiv'],\
                    thispdf['Activity ID']))
              print('Expected Cts: {}\tExpected Cts Err: {}\tTotalHitEff: {}'.format( \
                                        thispdf['Expected Counts'],\
                                        thispdf['Expected Counts Err'],\
                                        thispdf['TotalHitEff_K']))



            if self.components.empty:
               self.components = pd.DataFrame(columns=thispdf.index.values)
            self.components.loc[index] = thispdf


            
              
                
    ##########################################################################
    # Reads the sheets from the .xlsx spreadsheet file into pandas dataframes.
    # The sheets with energy/fiducial bins are dicts of dicts of dataframs,
    # allowing us to index them as: data[<energy_bin>][<fid_vol>][C.V.]
    ##########################################################################
    def ReadBasicSheet(self, inSheetName ):
        
        print('\tReading %s...' % inSheetName)
        df = pd.read_excel( self.filename, sheet_name = inSheetName  ) 
   
        return df



    def ReadNestedSheet( self, inSheetName ):

        print('\tReading %s' % inSheetName)
        header_rows = 4

        # The data in the sheet can be read in directly by skipping the header.

        df = pd.read_excel( self.filename, sheet_name = inSheetName, header = header_rows ) 
        #for col in df.columns:
        #  print(col)
        # The header needs some massaging due to the way the 
        # Excel file is currently formatted.
        dfheader_tmp = pd.read_excel( self.filename, sheet_name = inSheetName, skipfooter = df.shape[0]+1 )
        dfheader_tmp = dfheader_tmp.T.reset_index()
        dfheader_tmp.columns = range(0,len(dfheader_tmp.columns))
        dfheader_tmp.columns = dfheader_tmp.loc[ dfheader_tmp[0] == 'Energy Range [keV]' ].iloc[0].tolist()
        dfheader_tmp.columns.name = ''
        dfheader = dfheader_tmp.reset_index(drop=True) # Reset index to start at 0
        index_of_col_headers_row = dfheader.index[ ~dfheader['Fiducial mass [tonne]'].isna() ].tolist()[0]
        index_of_first_data_row = index_of_col_headers_row + 1
        dfheader = dfheader.iloc[index_of_first_data_row:]
        dfheader = dfheader.reset_index(drop=True)  

        energy_ranges = dfheader.loc[ ~dfheader['Energy Range [keV]'].str.contains('Unnamed') ]['Energy Range [keV]'].tolist()
        fiducial_vols = dfheader.loc[ ~dfheader['Fiducial mass [tonne]'].isna() ]['Fiducial mass [tonne]'].tolist()

        unique_energy_ranges = list( set( energy_ranges ) )
        unique_fiducial_vols = list( set( fiducial_vols ) )

        n_vals_per_fid_vol = dfheader.loc[ ~dfheader['Fiducial mass [tonne]'].isna() ].index.values[1] - \
                             dfheader.loc[ ~dfheader['Fiducial mass [tonne]'].isna() ].index.values[0]
        n_fid_vol_per_en_bin = len(unique_fiducial_vols) 

        # Define column names for each dataframe. global_columns are the same in each frame,
        # while the local columns change depending energy and fiducial volume bins.
        global_columns = []
        num_global_columns = 0
        for colname in df.columns: 
            if colname == 'C.V.': break
            global_columns.append(colname) 
            num_global_columns += 1
        local_columns = []
        for colname in df.columns[num_global_columns:num_global_columns+n_vals_per_fid_vol]:
            local_columns.append(colname)

        # Actually fill the data from the sheet into a dict of dicts
        sheetData = {}
        for energy_range in unique_energy_ranges:
            #print('Energy Range: {}'.format(energy_range))
            if 'Full' in energy_range: continue # This one isn't used
            sheetData[energy_range] = {}
            for fiducial_vol in unique_fiducial_vols:
              #print('Fiducial Volume: {}'.format(fiducial_vol))
              df_tmp = pd.DataFrame() # Create an empty dataframe
              for column in global_columns:   # Add in the global columns 
                  #print('Column: {}'.format(column))
                  df_tmp[column] = df[column]
              for column in local_columns:    # Add in the local columns
                  #print('Local Column: {}'.format(column))
                  #print('n_vals_per_fid_vol: {}'.format(n_vals_per_fid_vol))
                  #print('n_fid_vol_per_en_bin: {}'.format(n_fid_vol_per_en_bin))
                  col_index = int( self.GetColIndex( energy_range,\
                                                     fiducial_vol,\
                                                     n_vals_per_fid_vol,\
                                                     n_fid_vol_per_en_bin,\
                                                     dfheader ) )
                  #print('col_index = {}'.format(col_index))
                  if int(col_index) > 0:
                    df_tmp[column] = df['%s.%s' % (column,col_index)]
                  elif int(col_index) == 0:
                    df_tmp[column] = df['%s' % column]
              sheetData[energy_range][str(fiducial_vol)] = df_tmp
                
        return sheetData

    ############################################################################
    # Gets the column name (assigned by pd.read_excel in the header files)
    # for a given fiducial mass and energy range
    ############################################################################
    def GetColIndex( self, energy_range, fiducial_volume, n_vals_per_fid_vol, n_fid_vol_per_energy_bin, dfheader ):

        # Get start and end indices for the given energy bin
        istart = dfheader.loc[ dfheader['Energy Range [keV]'] == energy_range ].index.values[0]
        iend   = istart + n_vals_per_fid_vol * n_fid_vol_per_energy_bin 
        df_subset = dfheader.iloc[istart:iend]

        # Using the above, get start index for relevant data values. If the spreadsheet doesn't contain
        # data for the relevant fiducial cut, return a -1, which will skip this entry in the sheet's dataframe.
        if len( df_subset.loc[ df_subset['Fiducial mass [tonne]'] == fiducial_volume ].index.values ) == 0:
            return -1
        jstart = df_subset.loc[ df_subset['Fiducial mass [tonne]'] == fiducial_volume ].index.values[0]

        # Convert the raw index into the pd.read_excel index given to the relevant column
        col_index = int( jstart ) / int( n_vals_per_fid_vol )
        return col_index

    ############################################################################
    # Gets the "name tag": the version, date, and other information contained
    # in the name of the excel file downloaded from the Materials DB
    # For instance, the nametag for '../tables/Summary_v68_2016-06-21_0nu.xlsx'
    # will be 'v68_2016-06-21_0nu'
    ############################################################################
    def GetExcelNameTag( self, namestring ):
        filename = namestring.split('/')[-1]
        filename_noext = filename.split('.')[0]
        nametag = '_'.join(filename_noext.split('_')[1:])
        return nametag 

