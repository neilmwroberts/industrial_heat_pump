##### Importing Libraries #####
# Libraries below are used to pull from for the Heat Pump model
from array import array
import math
import numpy as np
import numpy_financial as npf
import pandas as pd
import requests
import csv
import CoolProp
import yaml
from CoolProp.CoolProp import PropsSI 
from CoolProp.Plots import PropertyPlot
from CoolProp.Plots import SimpleCompressionCycle

from libraries import *
from refrigerant_properties import*
from utilities.unit_defs import ureg, Q_
# from uncertainties import ufloat as uf
# from uncertainties import unumpy as unp

## Note: Default values set to -1.0 need to be calculated and are initialized, but will 
## return an error if not calculated first.

##### Initialization #####
## This class calls the heat pump model and initializes it to dummy values.
class heat_pump:
    ##### Model Variables #####
    def __init__(self):
        ##### 1.COP #####
        ## Outputs
        self.ideal_COP = np.array([-1.0]*2)*ureg.dimensionless
        self.actual_COP = np.array([-1.0]*2)*ureg.dimensionless
        self.refrigerant = []

        ##### 2.Energy and Mass Flow #####
        ## Outputs
        self.cold_final_temperature = Q_(np.array([-1.0]*2), 'degC')
        self.power_in = Q_(np.array([-1.0]*2), 'kW') # Gives the Energy into the heat pump in power
        self.average_power_in = Q_('-1.0 kW')
        self.annual_energy_in = Q_('-1.0 MW*hr')
        
        ##### 3.System Sizing #####
        self.heat_pump_power = Q_('-1.0 kW')
        self.storage_size = Q_('-1.0 kWh')
        self.storage_volume = Q_('-1.0 liter')
        self.storage_pressure = Q_('-1.0 Pa')
        
        
        ##### 3.Heat Pump Costs #####
        ## Outputs
        self.heatpump_cost = Q_('-1.0 USD')
        self.storage_cost = Q_('-1.0 USD')
        self.year_one_energy_costs = Q_('-1.0 USD/yr')
        self.year_one_fixed_o_and_m = Q_('-1.0 USD/yr')
        self.year_one_variable_o_and_m = Q_('-1.0 USD/yr')
        self.year_one_operating_costs = Q_('-1.0 USD/yr')
        self.LCOH = Q_('-1.0 USD / MMMBtu')
        self.capacity_factor = Q_('-1.0')
        self.demand_charge_file = None

        self.n_hrs = 8760

        #self.construct_yaml_input_quantities('heatpump_model_inputs.yml')

    
    def construct_yaml_input_quantities(self, file_path):
        with open(file_path, "r") as file_desc:
            input_dict = yaml.safe_load(file_desc)

        for key in input_dict:
            var = input_dict[key]
            try:
                if not isinstance(var, dict):
                    continue
                else:
                    quant = Q_(var['val'], var['unit'])
                input_dict[key] = quant
            except KeyError:
                print('Something is wrong with input variable ' + key)
                quit()
        self.__dict__.update(input_dict)


    def make_input_quantity(self, input_yaml_str):
        input_dict = yaml.safe_load(input_yaml_str)
        for key in input_dict:
            var = input_dict[key]
            try:
                if not isinstance(var, dict):
                    continue
                else:
                    quant = Q_(var['val'], var['unit'])
                input_dict[key] = quant
            except KeyError:
                print('Something is wrong with input variable ' + key)
                quit()
        self.__dict__.update(input_dict)

    
    def mysum(self, array_or_float):
        try:
            if len(array_or_float.magnitude) > 1.0:
                return np.sum(array_or_float)
            else:
                return self.n_hrs*array_or_float
        except(TypeError):
            return self.n_hrs*array_or_float


    ## This subroutine within the heat pump class Initializes the heat pump to a process in the process library.
    ## This initialization is not essential as all values can be input individually, but this module is built to 
    ## simplify the building of the models.
    def initialize_heat_pump(self,sector,process_name):
        self.hot_temperature_desired = Q_(np.array([process[sector][process_name]['hot_temperature_desired']]*self.n_hrs), 'degC')
        self.hot_temperature_minimum = Q_(np.array([process[sector][process_name]['hot_temperature_minimum']]*self.n_hrs), 'degC')
        self.hot_specific_heat = Q_(working_fluid[process[sector][process_name]['hot_working_fluid']]['specific_heat'], 'kJ / kg / degK')
        self.cold_temperature_available = Q_(np.array([process[sector][process_name]['waste_temperature']]*self.n_hrs), 'degC')

    ##### Model Calculations #####
    ## Calculating the COP
    def calculate_COP(self):
        
        # Calculating the ideal COP to begin with, this will be independent of the future anlaysis.
        self.ideal_COP = ((self.hot_temperature_desired.to('degK') + self.hot_buffer.to('degK')) )/((self.hot_temperature_desired.to('degK') + self.hot_buffer.to('degK')) - (self.cold_temperature_available.to('degK') - self.cold_buffer.to('degK')))
        
        
        if self.second_law_efficiency_flag == True:
            # If the carnot efficiency factor is true calculation the actual COP
            self.actual_COP = self.ideal_COP * self.second_law_efficiency
        else:
            # If the carnot efficiency factor is false requires more work in several steps
            # 1. If no refrigerant is selected pick one
            # 2. Using selected refrigerant calculate compressor efficiency
            # 3. Calculate actual COP from compressor efficiency
            # 4. Throw an error if calculation could not be completed.

            # Below will attempt to choose a refrigerant and calculate a realistic compressor efficiency from it, if this fails, it will revert to the carnot efficiency factor methodology
            ## Estimating Refrigerant Performance
            if self.refrigerant_flag != True:
                self.refrigerant = []
                for test_refrigerant in refrigerants:
                    t_crit = Q_(PropsSI(test_refrigerant, 'Tcrit'), 'kelvin').to('degC')
                    ## Checking if the refrigerant's critical temperature is at least 30°C > than the process temp.
                    if t_crit > (np.amax(self.hot_temperature_desired) + self.t_crit_delta):
                        self.refrigerant.append(test_refrigerant)
                
                print('Potential refrigerants include: ', self.refrigerant)
                ## Here the refrigerant with the lowest critical pressure, and therefore hopefully the lowest compression ratio
                ## is found and that will be recommended
                ## Need to update to reflect the fact that best refrigerant might not be the one with the lowest critical pressure
                min_p_crit = Q_('1e9 Pa')
                for test_refrigerant in self.refrigerant:
                    p_crit = Q_(PropsSI(test_refrigerant, 'Pcrit'), 'Pa')
                    if p_crit < min_p_crit:
                        min_p_crit = p_crit
                        self.refrigerant = test_refrigerant

            print('Selected refrigerant (based on user selection or minimual p_crit) is: ', self.refrigerant)

            ## Adjust such that this is below the Carnot Efficiency Factor 
            # Cycle calculation
            # Here the cycle points will be calculated. These points are:
            #  1. Compressor inlet
            #  2. Compressor outlet
            #  3. Expansion valve inlet
            #  4. Expansion valve outlet
            #  2-3 is the condenser where heat is expelled from the heat pump condenser to the heat sink or high temperature working fluid stream
            #  4-1 is the evaporator where heat is absorbed from the heat source or cold temperature working fluid to the heat pump evaporator
            self.refrigerant_high_temperature = self.hot_temperature_desired.to(ureg.degK) + self.hot_buffer.to(ureg.degK)
            self.refrigerant_low_temperature = self.cold_temperature_available.to(ureg.degK) - self.cold_buffer.to(ureg.degK)

            try:
                T_1 = np.array(self.refrigerant_low_temperature.m)
                T_3 = np.array(self.refrigerant_high_temperature.m)

                # Calculating Cycle Parameters
                P_1 = PropsSI('P', 'T', T_1, 'Q', 1, self.refrigerant)
                S_1 = PropsSI('S', 'T', T_1, 'Q', 1, self.refrigerant)
                H_1 = PropsSI('H', 'T', T_1, 'Q', 1, self.refrigerant)

                P_3 = PropsSI('P', 'T', T_3, 'Q', 0, self.refrigerant)
                S_3 = PropsSI('S', 'T', T_3, 'Q', 0, self.refrigerant)
                H_3 = PropsSI('H', 'T', T_3, 'Q', 0, self.refrigerant)

                T_2 = PropsSI('T', 'S', S_1, 'P', P_3, self.refrigerant)
                H_2 = PropsSI('H', 'S', S_1, 'P', P_3, self.refrigerant)

                P_2 = P_3
                H_2_prime = PropsSI('H', 'S', S_1, 'P', P_3, self.refrigerant)
                H_2 = H_1 + (H_2_prime - H_1)/(self.compressor_efficiency.m) # Remark, it should be tested if the state 2 (H_2, P_2) is in the 2-phase region or not
                T_2 = PropsSI('T', 'H', H_2, 'P', P_2, self.refrigerant)
                self.actual_COP = (np.divide((H_2 - H_3), (H_2 - H_1)))*ureg.dimensionless

                # There is an efficiency associated with the pressure ratio and an efficiency association with the volume ratio
                # The VR is taken from experimental values which we do not fully have, so will integrate as part of year 2
                # For now the VR is set to a constant value.
                # The compressor efficiency can also be set by the user
                # PR = P_2/P_1
                # eta_pr = 0.95-0.01*PR
                # eta_vr = 0.70
                # self.compressor_efficiency[i] = round(eta_vr*eta_pr, 3)
                # self.actual_COP = self.ideal_COP * self.compressor_efficiency

            except:
                print('There was an error calling refrigerant properties. Please check inputs and try again.')
                quit()

        if self.print_results: print('Calculate COP Called')
        if self.print_results: print('Average Theoretical COP: ', np.mean(self.ideal_COP))
        if self.print_results: print('Average Estimated COP: ', np.mean(self.actual_COP))

    ## Calculating working fluid energy and mass balance
    def calculate_energy_and_mass_flow(self):
        if self.print_results: print('Calculate Energy and Mass Called')

        # Initializing Temporary Arrays
        h_hi = Q_(np.array([-1.0]*self.n_hrs), 'J/kg')
        h_ho = Q_(np.array([-1.0]*self.n_hrs), 'J/kg')
        h_ci = Q_(np.array([-1.0]*self.n_hrs), 'J/kg')
        h_co = Q_(np.array([-1.0]*self.n_hrs), 'J/kg')

        # Converting MMBTU to kWh/hr (as it is expressed for the full hours of the year)
        # self.process_heat_requirement_kw = self.process_heat_requirement.to(ureg.kW)
        
        # Calculating the Work into the heat pump
        self.power_in = self.process_heat_requirement.to('kW')/self.actual_COP
        #for i in range(0,8760):
        #    self.power_in[i] = self.process_heat_requirement_kw[i]/self.actual_COP
        self.average_power_in = np.mean(self.power_in)
        self.annual_energy_in = self.mysum(self.power_in*Q_('1 hour')).to('kWh')

        
        # Calculating the Hot and Cold Mass Flow Parameters
        ## Hot
        h_hi = Q_(PropsSI('H', 'T', self.hot_temperature_minimum.to('degK').m, 'P', self.hot_pressure.to('Pa').m, self.hot_refrigerant), 'J/kg')
        h_ho = Q_(PropsSI('H', 'T', self.hot_temperature_desired.to('degK').m, 'P', self.hot_pressure.to('Pa').m, self.hot_refrigerant), 'J/kg')
        try:
            #if (self.hot_mass_flowrate == None) and (self.process_heat_requirement != None):
            if (self.hot_mass_flowrate == None):
                self.hot_mass_flowrate = (self.process_heat_requirement.to('W')/(h_ho - h_hi)).to('kg/s')
            else:
                self.process_heat_requirement = (self.hot_mass_flowrate.to('kg/s')*(h_ho - h_hi)).to('kW')
        except:
            print('Provide either .hot_mass_flowrate or .process_heat_requirement.')
            quit()

        ## Cold
        #cold_dT_array = self.cold_buffer - self.cold_deltaT #NR - commented this out. it's mixing quantities

        h_ci = Q_(PropsSI('H', 'T', self.cold_temperature_available.to('degK').m, 'P', self.cold_pressure.to('Pa').m, self.cold_refrigerant), 'J/kg')
        self.cold_final_temperature = self.cold_temperature_available - self.cold_deltaT
    

        #self.cold_final_temperature = self.cold_temperature_available - cold_dT_array #NR commented this out too since cold_DT_array was calculated from the heat exchanger approach temp (aka buffer)
        h_co = Q_(PropsSI('H', 'T', self.cold_final_temperature.to('degK').m, 'P', self.cold_pressure.to('Pa').m, self.cold_refrigerant), 'J/kg')
        self.cold_mass_flowrate = (self.process_heat_requirement.to('W')-self.power_in.to('W'))/(h_ci - h_co)
        
        
        # Getting average values for reporting
        self.hot_mass_flowrate_average = np.mean(self.hot_mass_flowrate).to('kg /s')
        
        if self.print_results: 
            print('Hot Mass Flow Average: {:~.3P}'.format(self.hot_mass_flowrate_average))
            print('Cold Average Outlet Temperature: {:~.2fP}'.format(np.mean(self.cold_final_temperature)))
            print('Cold Mass Flow Rate:  {:~.2fP}'.format(np.mean(self.cold_mass_flowrate.to('kg/s'))))

        # Calculating the Work into the heat pump
        self.power_in = self.process_heat_requirement.to('kW')/self.actual_COP
        #for i in range(0,8760):
        #    self.power_in[i] = self.process_heat_requirement_kw[i]/self.actual_COP
        self.average_power_in = np.mean(self.power_in)
        self.annual_energy_in = self.mysum(self.power_in*Q_('1 hour')).to('kWh')

        
        if self.print_results: 
            print('Average Power Draw of Heat Pump: {:~.3fP}'.format(self.average_power_in))
            print('Maximum Power Draw of Heat Pump: {:~.3fP}'.format(np.amax(self.power_in)))
            print('Annual Electricity in: {:,~.1fP}'.format(self.annual_energy_in))
        

#calculate demand charges
    def calculate_tou_demand_charges(self, demand_csv_path):
        
        df = pd.read_csv(demand_csv_path)
        df['datetime'] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index('datetime')
    
        assert len(df) == len(self.power_in), "Power and rate time series must be same length"
    
        power_series = pd.Series(self.average_power_in.magnitude, index=df.index) #2025-07-18 NR for now we assume there is enough storage to smooth the load
    
        total_charge = Q_(0.0, 'USD')
    
        for period, col in [
            ('facilities', 'facilities_related_demand'),
            ('on_peak', 'on_peak_demand_charge'),
            ('mid_peak', 'mid_peak_demand_charge')
        ]:
            # For each month
            monthly_charge = 0.0
            for month, df_month in df.groupby(df.index.to_period("M")):
                # Keep only hours where this demand charge applies
                applicable = df_month[df_month[col] > 0]
                if applicable.empty:
                    continue
    
                # Corresponding power during those hours
                month_power = power_series[applicable.index]
    
                # Find the hour with max power among applicable times
                peak_hour = month_power.idxmax()
                peak_kw = power_series.loc[peak_hour]
                demand_rate = df.loc[peak_hour, col]
    
                monthly_charge += peak_kw * demand_rate
    
            total_charge += Q_(monthly_charge, 'USD')
    
        return total_charge

    def calculate_LCOH(self, capital_costs, operating_costs):
        capex = capital_costs.to('USD')
        print(capex)
        annual_operating_costs = operating_costs.to('USD/yr')
    
        N = self.lifetime.to('yr').magnitude
        r = self.discount_rate.to('dimensionless').magnitude
        crf = r * (1 + r)**N / ((1 + r)**N - 1)
    
        # annualized values:
        annualized_capex = capex * crf / Q_('1 yr')
        total_annual_cost = (annualized_capex + annual_operating_costs) * Q_("1 year")  # USD
        
        # annual heat output as a flow per year
        annual_heat_output = self.mysum(self.process_heat_requirement.to('MMBtu/hr') * Q_('1 hr')).to('MMBtu')
    
        LCOH = (total_annual_cost / annual_heat_output).to('USD/MMBtu')
    
        if self.print_results:
            print(f"LCOH: {LCOH:~.2fP}")
    
        return LCOH

        
        
        ## Calculating Heat Pump Costs
    def calculate_heat_pump_costs(self):
        if self.print_results:
            print('Calculate Heat Pump Costs')
    
            # ---- convert process_heat_requirement to kW as a pandas Series/array ----
            # Assuming self.process_heat_requirement is a Pint Quantity array with hourly data
            load_kw = self.process_heat_requirement.to('kW').magnitude  # strip units for rolling
            load_series = pd.Series(load_kw)
        
            # ---- compute rolling averages ----
            # 7-day (weekly) moving average of hourly data (window=168 hours)
            weekly_moving_avg = load_series.rolling(window=168, min_periods=1).mean()
            max_weekly_avg = weekly_moving_avg.max()  # in kW (thermal)
        
            # 1-day (daily) moving average of hourly data (window=24 hours)
            daily_moving_avg = load_series.rolling(window=24, min_periods=1).mean()
            max_daily_avg = daily_moving_avg.max()  # in kW (thermal)
        
            # ---- size equipment based on your rules ----
            # Heat pump power = 3 × highest weekly moving average so we can go down to 33% duty cycle even in the worst week
            heat_pump_power_kw = 3 * max_weekly_avg
        
            # Storage capacity = 2 × highest daily moving average × hours in a day
            # (daily average in kW × 24h = daily energy in kWh)
            storage_kwh = 2.0 * max_daily_avg * 24.0
        
            # store them as Pint Quantities
            self.heat_pump_power = Q_(heat_pump_power_kw, 'kW')
            self.storage_size = Q_(storage_kwh, 'kWh')

            ####Calculate storage volume and pressure
            fluid = 'Water'

            # Get enthalpies at process and storage temperatures
            h_hot = Q_(PropsSI('H', 'T', self.storage_temperature.to('K').m, 'Q', 0, fluid), 'J/kg')
            h_cold = Q_(PropsSI('H', 'T', self.process_temperature.to('K').m, 'Q', 0, fluid), 'J/kg')
            
            # Thermal energy per kg between those temps:
            delta_h = (h_hot - h_cold).to('kJ/kg')  # kJ per kg
            
            # Convert your storage_size (kWh) to kJ
            stored_energy = self.storage_size.to('kJ')  # kJ total
            
            # Required mass of fluid:
            mass_needed = stored_energy / delta_h  # result in kg
            
            # Density at storage temperature (liquid water):
            rho = Q_(PropsSI('D', 'T', self.storage_temperature.to('K').m, 'Q', 0, fluid), 'kg/m^3')
            
            # Volume in m^3 and L:
            volume_m3 = mass_needed / rho
            volume_L = volume_m3.to('liter')
            
            # Saturation pressure at storage temperature (for vessel rating):
            p_sat = Q_(PropsSI('P', 'T', self.storage_temperature.to('K').m, 'Q', 0, fluid), 'Pa')
            
            # Store for later use
            self.storage_volume = volume_L
            self.storage_pressure = p_sat
            
            # Example prints:
            if self.print_results:
                print(f"Storage Volume: {volume_L:~.2fP}")
                print(f"Saturation Pressure: {p_sat.to('bar'):~.2fP}")
        
            # ---- cost calculations ----
            # Capital cost based on installed heat pump power and storage
            self.heatpump_cost = self.specific_heatpump_cost * self.heat_pump_power
            self.storage_cost = self.specific_storage_cost * self.storage_volume
            self.discharge_cost=self.specific_discharge_cost * max(self.process_heat_requirement)
            self.total_capital=self.heatpump_cost+self.storage_cost+self.discharge_cost
        
            # Year one fixed O&M cost based on peak capacity (thermal)
            self.year_one_fixed_o_and_m = self.fixed_o_and_m_per_size * self.heat_pump_power
            #removed variable O/M - we don't have any decent reference for this - better to not include
        
            
            # Calculating the Capacity Factor
            self.capacity_factor = self.mysum(self.process_heat_requirement.to('kW'))/(self.n_hrs*np.max(self.process_heat_requirement.to('kW')))
    
            # Calculating the kWh costs
            kwh_costs = Q_(np.array([0.0]*self.n_hrs), 'USD')
            #kwh_costs = self.hourly_utility_rate*self.power_in*Q_('1 hr') NR commented out
            kwh_costs = (self.hourly_utility_rate * self.average_power_in * Q_('1 hr')).to('USD') #changed to average power for now because we don't have a loop that generates hour by hour electrical power given a system storage capacity - this is fair for relatively large storage capacities
    
            #calculate demand charges
            kw_costs = self.calculate_tou_demand_charges(self.demand_charge_file)
            #original
            #kw_costs = 12*self.utility_rate*np.amax(self.power_in) # What is this 12? What are the units?
    
            self.year_one_energy_costs = (np.sum(kwh_costs)+kw_costs)/Q_('1 yr')
            self.year_one_operating_costs = self.year_one_fixed_o_and_m + self.year_one_energy_costs
    
            # kwh_costs is a vector of USD per hour, summed gives USD total
            annual_energy_cost_total = (np.sum(kwh_costs) + kw_costs)

            # convert total USD to a rate USD/year
            annual_energy_costs = annual_energy_cost_total / Q_("1 year")

            self.LCOH = self.calculate_LCOH(self.heatpump_cost + self.storage_cost + self.discharge_cost,self.year_one_operating_costs)

    
            if self.print_results: 
                print('System Power: {:,~.2fP}'.format(self.heat_pump_power))
                print('System Storage: {:,~.2fP}'.format(self.storage_size))            
                print('Heat Pump Cost: {:,~.2fP}'.format(self.heatpump_cost))
                print('Storage Cost: {:,~.2fP}'.format(self.storage_cost))
                print('Steam Dischage Cost: {:,~.2fP}'.format(self.discharge_cost))
                print('Total Capital Cost: {:,~.2fP}'.format(self.total_capital))
                print('Capacity Factor: {:~.3fP}'.format(self.capacity_factor))
                print('One Year Fixed O&M Costs: {:,~.2fP}'.format(self.year_one_fixed_o_and_m))
                print('One Year Generation&Transmission Costs: {:,~.2fP}'.format(np.sum(kwh_costs)))
                print('Demand Chargess:  {:,~.2fP}'.format(kw_costs)) ## this line added
                print('One Year Energy Costs: {:,~.2fP}'.format(self.year_one_energy_costs))
                print('One Year Operating Costs: {:,~.2fP}'.format(self.year_one_operating_costs))
                print('Lifetime LCOH: {:,~.2fP}'.format(self.LCOH))

    def write_output(self, filename):
        data = [
            ['Cold Temperature Available', '{:~.2fP}'.format(self.cold_temperature_available)],
            ['Cold Temperature Final', '{:~.2fP}'.format(self.cold_final_temperature)],
            ['Cold Mass Flowrate', '{:~.3fP}'.format(np.mean(self.cold_mass_flowrate).to('kg / s'))],
            ['Hot Temperature Desired', '{:~.2fP}'.format(self.hot_temperature_desired)],
            ['Hot Temperature Minimum', '{:~.2fP}'.format(self.hot_temperature_minimum)],
            ['Hot Mass Flowrate', '{:~.3fP}'.format(self.hot_mass_flowrate_average)],
            ['Ideal COP Calculated', '{:~.3fP}'.format(self.ideal_COP)],
            ['Selected Refrigerant', self.refrigerant],
            ['Estimated Compressor Efficiency', '{:~.3fP}'.format(self.compressor_efficiency)],
            ['Second Law Efficiency', '{:~.3fP}'.format(self.second_law_efficiency)],
            ['Carnot Efficiency Factor Flag ', self.second_law_efficiency_flag],
            ['Actual COP Calculated', '{:~.3fP}'.format(self.actual_COP)],
            ['Process Heat Average', '{:~.2fP}'.format(np.mean(self.process_heat_requirement.to('MMBtu/hr')))],
            ['Process Heat Average', '{:~.2fP}'.format(np.mean(self.process_heat_requirement.to('kW')))],
            ['Utility Rate Average', '{:,~.2fP}'.format(np.mean(self.hourly_utility_rate))],
            ['Capacity Factor', '{:~.3fP}'.format(np.mean(self.capacity_factor))],
            ['Project Lifetime', '{:~.2fP}'.format(self.lifetime)],
            ['Power in Average', '{:~.2fP}'.format(self.average_power_in)],
            ['Annual Energy In', '{:~.2fP}'.format(self.annual_energy_in)],
            ['HP Capital Cost Per Unit', '{:,~.2fP}'.format(self.specific_heatpump_cost)],
            ['Storage Capital Cost Per Unit', '{:,~.2fP}'.format(self.specific_storage_cost)],
            ['Fixed O&M Costs', '{:,~.2fP}'.format(self.fixed_o_and_m_per_size)],
            ['Heat Pump Thermal Power', '{:,~.2fP}'.format(self.heat_pump_power)],
            ['Thermal Storage', '{:,~.2fP}'.format(self.storage_size)],
            ['Heat Pump Capital Cost', '{:,~.2fP}'.format(self.heatpump_cost)],
            ['Storage Capital Cost', '{:,~.2fP}'.format(self.storage_cost)],
            ['Total Capital Cost', '{:,~.2fP}'.format(self.total_capital)],
            ['Year One Energy Costs', '{:,~.2fP}'.format(self.year_one_energy_costs)],
            ['Year One Fixed O&M Costs', '{:,~.2fP}'.format(self.year_one_fixed_o_and_m)],
            ['Year One Variable O&M Costs', '{:,~.2fP}'.format(self.year_one_variable_o_and_m)],
            ['Year One Total Operating Costs', '{:,~.2fP}'.format(self.year_one_operating_costs)],
            ['LCOH', '{:,~.2fP}'.format(self.LCOH)],
            ]
        
        df_output = pd.DataFrame(data,columns=['Variable','Value'])
        df_output.to_csv('output/'+filename+'.csv')
        if self.print_results: print('Writing all output to a file')

    def run_all(self,filename):
        self.calculate_COP()
        self.calculate_energy_and_mass_flow()
        self.calculate_heat_pump_costs()
        if self.write_output_file: self.write_output(filename)

