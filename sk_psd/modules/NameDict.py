########################################################################
##  
## This is a class to map names of components in the spreadsheet
## to names of components in the rootfile names. This should be a 
## temporary thing...
##      - B. Lenardo (7.16.2018)
##
#######################################################################



class NameDict:

  def __init__(self):
    self.data = {}
    self.data["OuterCryostatResin"] = 'Outer Cryostat (Resin)'
    self.data["OuterCryostatFiber"] = 'Outer Cryostat (Fiber)'
    self.data["OuterCryostatSupportResin"] = 'Outer Cryostat Support (Resin)'
    self.data["OuterCryostatSupportFiber"] = 'Outer Cryostat Support (Fiber)'
    self.data["InnerCryostatResin"] = 'Inner Cryostat (Resin)'
    self.data["InnerCryostatFiber"] = 'Inner Cryostat (Fiber)'
    self.data["InnerCryostatSupportResin"] = 'Inner Cryostat Support (Resin)'
    self.data["InnerCryostatSupportFiber"] = 'Inner Cryostat Support (Fiber)'
    self.data["InnerCryostatLiner"] = 'Inner Cryostat Liner'
    self.data["HFE"] = 'HFE'
    self.data["HVTubes"] = 'HV Tubes'
    self.data["HVCables"] = 'HV Cables'
    self.data["HVFeedthrough"] = 'HV Feedthrough'
    self.data["HVFeedthroughCore"] = 'HV Feedthrough Core'
    self.data["HVPlunger"] = 'HV Plunger'
    self.data["CalibrationGuideTube1"] = 'Calibration Guide Tube 1'
    self.data["CalibrationGuideTube2"] = 'Calibration Guide Tube 2'
    self.data["TPCVessel"] = 'TPC Vessel'
    self.data["TPCSupportCone"] = 'TPC Support Cone'
    self.data["CathodeRadon"] = 'Cathode (Radon)'
    self.data["Cathode"] = 'Cathode'
    self.data["Bulge"] = 'Bulge'
    self.data["FieldRings"] = 'Field Rings'
    self.data["SupportRodsandSpacers"] = 'Support Rods and Spacers'
    self.data["SiPMStaves"] = 'SiPM Staves'
    self.data["SiPMModuleInterposer"] = 'SiPM Module (Interposer)'
    self.data["SiPMElectronics"] = 'SiPM Electronics'
    self.data["SiPMCables"] = 'SiPM Cables'
    self.data["SiPMs"] = 'SiPMs'
    self.data["ChargeModuleCables"] = 'Charge Tiles Cables'
    self.data["ChargeModuleElectronics"] = 'Charge Tiles Electronics'
    self.data["ChargeModuleSupport"] = 'Charge Tiles Support'
    self.data["ChargeModuleBacking"] = 'Charge Tiles Backing'
    self.data["FullLXe"] = 'Full LXe'
    #        self.data["bb2n"] = 'Full LXe'
    #        self.data["bb0n"] = 'Full LXe'
    self.data["ActiveLXe"] = 'Active LXe'
    self.data["InactiveLXe"] = 'Inactive LXe'
    self.data["SolderAnode"] = 'Solder (Anode)'
    self.data["SolderSiPM"] = 'Solder (SiPM)'
