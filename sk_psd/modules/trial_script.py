import nEXOFitWorkspace


y = nEXOFitWorkspace.nEXOFitWorkspace()
#y.LoadInputDataframe('../tables/Summary_D-005_v22_2018-11-12.xls')
#y.LoadInputDataframe('../tables/SimulationPDFs_D-005_v22_2018-11-12_02_02.h5')
#y.LoadInputDataframe('../tables/Summary_D-005_v22_2018-11-12_02.h5')
y.LoadInputDataframe('../tables/Summary_D-005_v22_2018-11-12_02.h5')

y.CreateGroupedPDFs()

