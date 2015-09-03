from collections import defaultdict
import re

import constants
import pickle

import numpy

import os.path

import scipy.constants as sc


"""Constants used commonly in solcore
"""

class UnitsLibrary():
    def load(self):
        u = UnitsSystem()

    def unload(self):
        u = UnitsSystem()
        UnitsSystem.read()
        

class UnitError(Exception):
    def __init__(self, msg):
        BaseException.__init__(self, msg)

class WrongDimensionError(Exception):
    def __init__(self, msg):
        BaseException.__init__(self, msg)


def generateConversionDictForSISuffix(suffix, centi = False, deci = False, non_base_si_factor=1):
    prefixes = "Y,Z,E,P,T,G,M,k,,m,u,n,p,f,a,z,y".split(",")
    exponents = list(range(8,-9,-1))
    
    if centi:
        prefixes.append("c")
        exponents.append(-2./3.)
        
    if deci:
        prefixes.append("d")
        exponents.append(-1./3.)

    
    unitNames = ["%s%s"%(prefix,suffix) for prefix in prefixes]
    conversion =[1000.**exponent*non_base_si_factor for exponent in exponents]
    
    return dict(zip(unitNames, conversion))

class UnitsSystem():
    def __init__(self):
        self.separate_value_and_unit_RE = re.compile(u"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)(?:[ \t]*(.*))?")
        self.split_units_RE = re.compile(u"(?:([^ \+\-\^\.0-9]+)[\^]?([\-\+]?[^ \-\+]*)?)")
        self.siConversions = {}
        self.dimensions = defaultdict(dict)
        self.read(name="defaultunits")

        
    def read(self, name=None):
        self.read_database()
        self.si_conversions = {}
        for dimension in self.database.sections():
            units = self.database.options(dimension)
            for unit in units:
                if "META_GenerateConversions" in unit:
                    expression = self.database.get(dimension, unit)
                    si_base_unit = expression.split()[0]
                    centi = "centi" in expression
                    deci = "deci" in expression
                    non_base_si_factor = self.safe_eval(expression.split()[1]) if "altbase" in expression else 1
                    dimension_conversions = generateConversionDictForSISuffix( 
                        si_base_unit, 
                        centi = centi,
                        deci=deci,
                        non_base_si_factor = non_base_si_factor
                    )
                    self.siConversions.update(dimension_conversions)
                    self.dimensions[dimension].update(dimension_conversions)
                    continue
                string_expression = self.database.get(dimension,unit)
                self.siConversions[unit] = self.safe_eval(string_expression)
                self.dimensions[dimension][unit]= self.siConversions[unit]

    def read_database(self):
        work_dir=os.path.dirname(os.path.realpath(__file__))
        fp=open(work_dir+"/databasedump.pkl","rb")
        self.database=pickle.load(fp)
        fp.close()

    def write_database(self):
        fp=open("databasedump.pkl","wb")
        pickle.dump(self.database,fp)
        fp.close()

    def safe_eval(self, string_expression):
        return eval(string_expression,{"__builtins__":{}},{"constants":constants})
    
    def siUnits(self, value, unit):
        """convert value from unit to equivalent si-unit
        >>> print siUnits(1,"mm") # yields meters
        0.001
        >>> print siUnits(1,"um") # yields meters
        1e-06
        """
        if unit is None or value is None:
            return value
    
        units_list = self.split_units_RE.findall(unit)
        for unit, power in units_list:
            power = float(power) if power != '' else 1
            value = value * numpy.power((self.siConversions[unit]),power) ### caution, *= is WRONG because it modifies original obj. DO NOT WANT

        return value
    def asUnit(self, value, unit, dimension=None):
        """ converts from si unit to other unit
        >>> print asUnit(1, "mA") # print 1A in mA.
        1000.0
        """
        if unit is None or value is None:
            return value
            
        units_list = self.split_units_RE.findall(unit)
        for unit, power in units_list:
            power = float(power) if power != '' else 1
            value = value / (self.siConversions[unit])**power ### caution, /= is WRONG because it modifies original obj. DO NOT WANT

        return value
    def si(self, *args):
        """ Utility function that forwards to either siUnit or siUnitFromString"""
        if type(args[0]) == str:
            return self.siUnitFromString(*args)
        return self.siUnits(*args)
    def siUnitFromString(self,string):
        """ converts a string of a number with units into si units of that quantity

        >>> print si("5 mm s-1") # output in m/s
        0.005

        >>> print si("5e-0mm-2") # output in m2
        5000000.0
    
        >>> print si("5")
        5.0

        """
        # if unit is None or value is None:
        #     return value
        
        matchObj = self.separate_value_and_unit_RE.match(string)
        value, unit = matchObj.groups()
        value = float(value)
        units_list = self.split_units_RE.findall(unit)
        for unit, power in units_list:
            power = float(power) if power != '' else 1
            value *= (self.siConversions[unit])**power
        return value
    
    def convert(self, value, from_unit, to_unit):
        """ converts between comparable units, does NOT check if units are comparable.
        >>> print convert(1, "nm", "mm")
        1e-06
        >>> print convert(1, "um", "nm")
        1000.0
        >>> print convert(1, "cm s-1", "km h-1")
        0.036
        """
        return self.asUnit(self.siUnits(value, from_unit), to_unit)
    def eVnm(self,value):
        """ Bi-directional conversion between nm and eV.
    
        Arguments:
        value -- A number with units [nm] or [eV].
    
        Returns:
        Either the conversion [nm] --> [eV], or [eV] --> [nm]
    
        >>> print '%.3f'%eVnm(1000)
        1.240
        >>> print '%i'%round(eVnm(1))
        1240
        """
        factor = self.asUnit(sc.h, "eV") * self.asUnit(sc.c, "nm")
        return factor / value
    def nmJ(self,value):
        """ Bi-directional conversion between nm and eV.
    
        Arguments:
        value -- A number with units [nm] or [eV].
    
        Returns:
        Either the conversion [nm] --> [eV], or [eV] --> [nm]
    
        >>> print '%.3f'%eVnm(1000)
        1.240
        >>> print '%i'%round(eVnm(1))
        1240
        """
        factor = sc.h  * sc.c
        return factor / self.siUnits(value,"nm")
    def mJ(self,value):
        """ Bi-directional conversion between nm and eV.
    
        Arguments:
        value -- A number with units [nm] or [eV].
    
        Returns:
        Either the conversion [nm] --> [eV], or [eV] --> [nm]
    
        >>> print '%.3f'%eVnm(1000)
        1.240
        >>> print '%i'%round(eVnm(1))
        1240
        """
        factor = sc.h  * sc.c
        return factor / value

    def nmHz(self, value):
        """Bi-directional conversion between nm and Hz.
    
        Arguments:
        value -- A number with units [nm] or [Hz].
    
        Returns:
        Either a number which is the conversion [nm] --> [Hz] or [Hz] --> [nm]
        """
        factor = self.asUnit(sc.c, "nm s-1")
        return factor / value
    def spectral_conversion_nm_ev(self, x, y):
        """Bi-directional conversion between a spectrum per nanometer and a spectrum per electronvolt.
    
        Arguments:
        x -- abscissa of the spectrum in units of [nm] or [eV]
        y -- ordinate of the spectrum in units of [something/nm] or [something/eV]
    
        Returns:
        A tuple (x, y) which has units either [eV, something/eV] or [nm. something/nm].
    
        Example:
        1) nm --> eV conversion
        wavelength_nm
        photon_flux_per_nm
        energy_ev, photon_flux_per_ev = spectral_conversion_nm_ev(wavelength_nm, photon_flux_per_nm)
    
        2) eV --> nm conversion
        energy_ev
        photon_flux_per_ev
        wavelength_nm, photon_flux_per_nm = spectral_conversion_nm_ev(energy_ev, photon_flux_per_ev)

        Discussion:
        A physical quantities such as total number of photon in a spectrum or 
        total energy of a spectrum should remain invariant after a transformation
        to different units. This is called a spectral conversion. This function
        is bi-directional because the mathematics of the conversion processes
        is symmetrical.
    
        >>> import numpy as np 
        >>> x = np.array([1,2,3])
        >>> y = np.array([1,1,1])
        >>> area_before = np.trapz(y, x=x)
        >>> x_new, y_new = spectral_conversion_nm_ev(x, y)
        >>> area_after = np.trapz(y_new, x=x_new)
        >>> compare_floats(area_before, area_after, relative_precision=0.2)
        True
        """
        x_prime = self.eVnm(x)
        conversion_constant = self.asUnit(sc.h, "eV s") * self.asUnit(sc.c, "nm s-1")
        y_prime = y * conversion_constant / x_prime**2
        y_prime = reverse(y_prime) # Wavelength ascends as electronvolts decends therefore reverse arrays
        x_prime = reverse(x_prime)
        return (x_prime, y_prime)
    def spectral_conversion_nm_hz(self, x, y):
        """Bi-directional conversion between a spectrum per nanometer and a spectrum per Hertz.
    
        Arguments:
        x -- abscissa of the spectrum in units of [nm] or [Hz]
        y -- ordinate of the spectrum in units of [something/nm] or [something/Hz]
    
        Returns:
        A tuple (x, y) which has units either [eV, something/nm] or [nm. something/Hz].
    
        Example:
        1) nm --> Hz conversion
        wavelength_nm
        photon_flux_per_nm
        energy_hz, photon_flux_per_hz = spectral_conversion_nm_hz(wavelength_nm, photon_flux_per_nm)
    
        2) Hz --> nm conversion
        energy_hz
        photon_flux_per_hz
        wavelength_nm, photon_flux_per_nm = spectral_conversion_nm_ev(energy_hz, photon_flux_per_hz)

        Discussion:
        A physical quantities such as total number of photon in a spectrum or 
        total energy of a spectrum should remain invariant after a transformation
        to different units. This is called a spectral conversion. This function
        is bi-directional because the mathematics of the conversion processes
        is symmetrical.
    
        >>> import numpy as np 
        >>> x = np.array([1,2,3])
        >>> y = np.array([1,1,1])
        >>> area_before = np.trapz(y, x=x)
        >>> x_new, y_new = spectral_conversion_nm_hz(x, y)
        >>> area_after = np.trapz(y_new, x=x_new)
        >>> compare_floats(area_before, area_after, relative_precision=0.2)
        True
        """
        x_prime = self.nmHz(x)
        conversion_constant = self.asUnit(sc.c, "nm s-1")
        y_prime = y * conversion_constant / x_prime**2
        y_prime = reverse(y_prime) # Wavelength ascends as frequency decends therefore reverse arrays
        x_prime = reverse(x_prime)
        return (x_prime, y_prime)
    def sensibleUnits(self, value, dimension, precision=2):
        """ attempt to convert a physical quantity of a particular dimension to the most sensible units
        >>> print sensibleUnits(0.001,"length",0)
        1 mm
        >>> print sensibleUnits(1000,"length",0)
        1 km
        >>> print sensibleUnits(si("0.141 days"),"time", 5)
        3.38400 h
        """
    
        negative = ""
        if value <0: 
            value*=-1
            negative = "-"
        formatting = "%s%%.%if %%s"%(negative,precision)
        d= dimensions[dimension]
        possibleUnits = d.keys()
        if value == 0:
            return formatting%(0, "")
        allValues = [abs(log10(asUnit(value, unit))) for unit in possibleUnits]
        bestUnit = possibleUnits[allValues.index(min(allValues))]
        return formatting%(asUnit(value, bestUnit), bestUnit)
    def eV(self, e,f=3):
        return "%.3f eV"%self.asUnit(e,"eV")
    def guess_dimension(self, unit):
        """
        >>> print guess_dimension("nm")
        length
        """
        possibilities = [key for key in self.dimensions.keys() if unit in self.dimensions[key]]
    
        assert len(possibilities) != 0, "Guessing dimension of '%s': No candidates found"%unit
        assert len(possibilities) == 1, "Guessing dimension of '%s': Multiple candidates found, please convert manually. (%s)"%(unit, ", ".join(possibilities))
    
        return possibilities[0]

    def list_dimensions(self):
        for dim in self.dimensions.keys():
            print ("%s: %s"%(dim, ", ".join([k for k in self.dimensions[dim].keys() if k is not None and k is not ""])))


    # def __repr__(self):
    #     print (compare_floats)
    #     return "Moo"
def compare_floats(a, b, absoulte_precision=1e-12, relative_precision=None):
    """Returns true if the absolute difference between the numbers a and b is less than the precision.

    Arguments:
    a -- a float
    b -- a float

    Keyword Arguments (optional):
    absolute_precision -- the absolute precision, abs(a-b) of the comparison.
    relative_precision -- the relative precision, max(a,b)/min(a,b) - 1. of the comparison.

    Returns:
    True if the numbers are the same within the limits of the precision.
    False if the number are not the same within the limits of the precision.
    """

    if relative_precision is None:
        absolute = abs(a-b)  
        if absolute < absoulte_precision:
            return True
        else:
            return False
    else:
        relative = max(a,b)/min(a,b) - 1.
        if relative < relative_precision:
            return True
        else:
            return False

def independent_nm_ev(x,y):
    return eVnm(x)[::-1],y[::-1]

def independent_nm_J(x,y):
    return nmJ(x)[::-1],y[::-1]

def independent_m_J(x,y):
    return reverse(mJ(x)),reverse(y)

def reverse(x):
    return x[::-1]
