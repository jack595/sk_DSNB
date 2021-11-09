import jsonschema

schemas = {}

# Monte Carlo
# -- rootfiles
schemas['rootfiles'] = {
  'type': 'array', 
  'items': {
    'type': 'object', 
    'title': 'ROOT files', 
    'properties': {
      'isotope': {'type': 'string', 'title': 'Isotope', 'enum':['U-238', 'Th-232', 'K-40', 'Co-60', 'Cs-137', 'Rn-222', 'Bi-214', 'Xe-137', 'Al-26', 'Ag-110m', 'Co-56', 'bb2n', 'bb0n', 'B8nu']},
      'numbersimed': {'type': 'string', 'title': 'Number of disintegrations simulated' },
      'filename': {'type': 'string', 'title': 'Name of attached ROOT file' },
      'datatype': {'type': 'string', 'title': 'Data type', 'enum': ['', 'histograms', 'tree']}
    },
    'required': ['isotope','numbersimed', 'filename', 'datatype'],
  }
}   
# -- main
schemas['montecarlo'] = {
  'type': 'object',
  'properties':{
    'title': {'type': 'string', 'title': 'Title' },
    'key': {'type': 'string', 'title': 'ID' },
    'component_name': {'type': 'string', 'title': 'Component'},
    'material': {'type': 'string', 'title': 'Material' },
    'mass': {'type': 'string', 'title': 'Mass' },
    'quantity': {'type': 'string', 'title': 'Quantity' },
    'remarks': {'type': 'string', 'title': 'Remarks' },
    'rootfiles': schemas['rootfiles'],
  },
  'required': ['title','key','component_name','material','mass','quantity','remarks','rootfiles'],
}

# Radioassay
# -- isotope
schemas['results_new'] = {
  'type': 'array',
  'items': {
    'type': 'object',  
    'title': 'Results by Isotopes', 
    'properties': {
      'isotope': {'type': 'string', 'title': 'Isotope', 'enum': ['', 'U-238', 'Th-232', 'K-40', 'Co-60', 'Cs-137', 'Rn-222', 'Bi-214', 'Xe-137', 'Al-26', 'Ag-110m', 'Co-56', 'bb2n', 'bb0n', 'B8nu']},
      'specific_activity': {'type': 'number', 'title': 'Specific Activity'},
      'error': {'type': 'number', 'title': 'Error'},
      'unit': {'type': 'string', 'title': 'Unit of measurement', 'enum': ['mBq/kg', 'ppb', 'ppt', 'ng/g', 'pg/g', 'mBq/cm&sup2;', 'ng/cm&sup2;','pg/cm&sup2;']},
      'error_type': {'type': 'string', 'title': 'Error type', 'enum': ['-','Symmetric error (68% C.L.)','Asymmetric error', 'Range', 'Upper limit (90% C.L.)', 'No errors specified']},
      'altlimit': {'type': 'number', 'title': 'Alternate limit'},
      'systerr': {'type': 'number', 'title': 'Systematic error'},
      'asym_lower_error':  {'type': 'number', 'title': 'Lower error (for asym. err.)'},
    },
    'required': ['isotope', 'specific_activity', 'error', 'unit', 'error_type', 'altlimit', 'systerr', 'asym_lower_error'],
  },
}
# -- analysis result
schemas['analysis_result'] = {
  'type': 'array', 
  'items': {
    'type': 'object', 
    'title': 'Analysis', 
    'properties': {
      'analyzer': {'type': 'string', 'title': 'Analyzer' },
      'quality': {'type': 'string', 'title': 'Analysis quality' , 'enum':['-','Recommended','Obsolete'], 'quality':''},
      'meas_type': {'type': 'string', 'title': 'Measurement type' , 'enum':['Regular measurement','Combined measurement']},
      'results_new': schemas['results_new'],
      'remarks': {'type': 'string', 'title': 'Analysis remarks' }
    },
    'required': [ 'analyzer', 'quality', 'meas_type', 'results_new', 'remarks'],
  }
}
# -- counting
schemas['counting'] = {
  'type': 'array', 
  'items': {
    'type': 'object', 
    'title': 'Counting', 
    'properties': {
      'institution': {'type': 'string', 'title': 'Institution', 
        'enum': ['', 'Center for Underground Physics (CUP)', 'EXO-200', 'GERDA', 'Institute of High Energy Physics (IHEP)',
                 'Laurentian University', 'Lawrence Livermore National Laboratory (LLNL)', 'LZ', 'National Institute of Standards and Technology (NIST)',
                 'National Research Council (NRC)', 'Pacific Northwest National Laboratory (PNNL)', 'Sanford Underground Research Facility (SURF)', 'SNOLAB', 'Stanford University',
                 'SURF', 'University of Alabama (UA)', 'University of Illinois', 'University of New Hampshire (UNH)', 
                 'University of Seoul', 'Vue-des-Alpes (VdA)', 'Other (state in counting remarks)']},
      'technique': {'type': 'string', 'title': 'Technique', 'enum': ['', 'Ge counting', 'ICP-MS', 'NAA', 'GD-MS', 'XPS', 'Other (state in counting remarks)']},
      'detector': {'type': 'string', 'title': 'Detector' },
      'measured_by': {'type': 'string', 'title': 'Measured by' },
      'date_of_measurement': {'type': 'date', 'title': 'Date of measurement (yyyy-mm-dd)' },
      'counted_from': {'type': 'date', 'title': 'Counted from (yyyy-mm-dd)' },
      'counted_to': {'type': 'date', 'title': 'Counted to (yyyy-mm-dd)' },
      'livetime': {'type': 'string', 'title': 'Livetime [s]' },
      'amount_new': {'type': 'number', 'title': 'Amount (Mass, area, or length)'},
      'unit_new': {'type': 'string', 'title': 'Unit', 'enum':[ 'piece(s)', 'kg', 'g', 'mg', 'lb', 'm', 'cm', 'mm', 'ft', 'in', 'm&sup2;', 'cm&sup2;', 'mm&sup2;', 'ft&sup2;', 'in&sup2;']},
      'descriptions': {'type': 'string', 'title': 'Counting descriptions' },
      'analysis_result': schemas['analysis_result'],
      'remarks': {'type': 'string', 'title': 'Counting remarks' },
    },
    'required': ['institution', 'technique', 'detector', 'measured_by', 'date_of_measurement', 
                 'counted_from', 'counted_to', 'livetime', 'amount_new', 'unit_new', 'descriptions', 'analysis_result', 'remarks']
  }
}
# -- samples
schemas['samples'] = {
  'type': 'array', 
  'items': {
    'type': 'object', 
    'title': 'Sample', 
    'properties': {
      'sample_id': {'type': 'string', 'title': 'Sample ID' },
      'supplier': {'type': 'string', 'title': 'Supplier' },
      'product': {'type': 'string', 'title': 'Product' },
      'part_number': {'type': 'string', 'title': 'Part number' },
      'lot_number': {'type': 'string', 'title': 'Lot number' },
      'amount_new': {'type': 'number', 'title': 'Amount (Mass, area, or length)'},
      'unit_new': {'type': 'string', 'title': 'Unit', 'enum': [ 'piece(s)', 'kg', 'g', 'mg', 'lb', 'm', 'cm', 'mm', 'ft', 'in', 'm&sup2;', 'cm&sup2;', 'mm&sup2;', 'ft&sup2;', 'in&sup2;']},
      'descriptions': {'type': 'string', 'title': 'Sample descriptions' },
      'tracking_quantity': {'type': 'string', 'title': 'Remaining quantity' },
      'tracking_unit': {'type': 'string', 'title': 'Unit (for "Remaining quantity")' },
      'tracking_location': {'type': 'string', 'title': 'Location' },
      'tracking_date': {'type': 'date', 'title': 'Date (yyyy-mm-dd)' },
      'tracking_mother_sample': {'type': 'string', 'title': 'Mother sample (leave blank if from vendor)' },
      'counting': schemas["counting"],
      'remarks': {'type': 'string', 'title': 'Sample remarks' }
    },
    'required': ['sample_id', 'supplier', 'product', 'part_number', 'lot_number', 'amount_new', 'unit_new', 'descriptions', 'tracking_quantity', 'tracking_unit', 'tracking_location',
                 'tracking_date', 'tracking_mother_sample', 'counting', 'remarks']
  }
}
# -- main
schemas['main'] = {
  'type': 'object',
  'properties': {
    'title': {'type': 'string', 'title': 'Material' },
    'key': {'type': 'string', 'title': 'ID' },
    'original_author':  {'type': 'string', 'title': 'Original Author' },
    'material_name': {'type': 'string', 'title': 'Material Name' },
    'intended_use': {'type': 'string', 'title': 'Intended use' },
    'actual_use': {'type': 'string', 'title': 'Actual use' },
    'real_or_mc': {'type': 'string', 'title': 'Real or MC', 'enum':['Real','MC']},
    'descriptions': {'type': 'textarea', 'title': 'Descriptions', 'default': '<pre>\n\n</pre>'},
    'samples': schemas['samples'],
    'remarks': {'type': 'textarea', 'title': 'Remarks' , 'default': '<pre>\n\n</pre>'},
  },
  'required': [ 'title', 'key', 'original_author', 'material_name', 'intended_use', 'actual_use', 'real_or_mc', 'descriptions', 'samples', 'remarks']
}

# Other fields
'''
    "author": "xenon", 
    "created_at": "2015-12-14T21:43:19.704Z", 
    "doctype": "montecarlo", 
    "last_edited_at": "2015-12-14T21:43:19.704Z", 
    "last_edited_by": "xenon", 
    "history": {}
'''

def validate(doc,doctype):
  schema = schemas[doctype]
  jsonschema.validate(doc, schema)
  
#doc = json.load(open('test.json','r'))
