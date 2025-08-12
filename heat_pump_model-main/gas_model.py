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

class gas_heater:
    def __init__(self):
        self.n_hrs = 8760
        ##### 2.Energy and Mass Flow #####
        ## Outputs
        ## Note keeping in line with heat pump model structure, the energy
        ## and mass flow is item number 2. Heat Pump item number 1 is COP 
        
        self.process_heat_requirement = Q_(np.array([-1.0]*2), 'kW')
        self.power_in = Q_(np.array([-1.0]*2), 'kW') # Gives the Energy into the heat pump in power
        self.thermal_efficiency = np.array([1.0]*self.n_hrs)*ureg.dimensionless
        self.average_power_in = Q_('-1.0 kW')
        self.annual_energy_in = Q_('-1.0 MW*hr')

        self.hot_temperature_out = Q_(np.array([90]*8760), 'degC')
        self.cold_temperature_in = Q_(np.array([60]*8760), 'degC')

        ##### 3.Gas Boiler costs #####
        self.capital_cost = Q_('-1.0 USD')
        self.year_one_energy_cost = Q_('-1.0 USD/yr')
        self.year_one_fixed_o_and_m = Q_('-1.0 USD/yr')
        self.year_one_variable_o_and_m = Q_('-1.0 USD/yr')
        self.year_one_operating_costs = Q_('-1.0 USD/yr')
        self.LCOH = Q_('-1.0 USD / MMMBtu')
        self.capacity_factor = Q_('-1.0')

        #self.construct_yaml_input_quantities('gas_model_inputs.yml')
    
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


    def calculate_LCOH(self, capital_costs, operating_costs):
        capex = capital_costs.to('USD')
        
        annual_operating_costs = operating_costs.to('USD/yr')
        
        N = self.lifetime.to('yr').magnitude
        r = self.discount_rate.to('dimensionless').magnitude
        crf = r * (1 + r)**N / ((1 + r)**N - 1) #capital recovery factor
        
        # annualized values:
        annualized_capex = capex * crf / Q_('1 yr')
        total_annual_cost = (annualized_capex + annual_operating_costs) * Q_("1 year")  # USD
        
        # annual heat output as a flow per year
        self.annual_heat_output = self.mysum(self.process_heat_requirement.to('MMBtu/hr') * Q_('1 hr')).to('MMBtu')
        
        LCOH = (total_annual_cost / self.annual_heat_output).to('USD/MMBtu')
        
        
        return LCOH

    
    def mysum(self, array_or_float):
        try:
            if len(array_or_float.magnitude) > 1.0:
                return np.sum(array_or_float)
            else:
                return self.n_hrs*array_or_float
        except(TypeError):
            return self.n_hrs*array_or_float

    def calculate_costs(self):

        self.peak_power = np.max(self.process_heat_requirement.to('kW'))
        self.capital_cost = self.specific_capital_cost * self.peak_power
        self.capital_cost = self.capital_cost.to('USD')
        
        # --- Fixed O&M (USD/yr) ---
        self.year_one_fixed_o_and_m = (self.fixed_o_and_m_per_size * self.peak_power.to('MMBtu/hr')) / Q_('1 yr')  # attach /yr
        
        # --- Variable O&M (USD/yr) ---
        self.year_one_variable_o_and_m = (self.variable_o_and_m * self.mysum(self.process_heat_requirement.to('MMBtu/hr') * Q_('1 hr'))) / Q_('1 yr')
        
        # --- Capacity factor (dimensionless) ---
        self.capacity_factor = (self.mysum(self.process_heat_requirement.to('kW') * Q_('1 hr')) /(self.n_hrs * np.max(self.process_heat_requirement.to('kW')) *Q_('1 hr'))).to('dimensionless')
        
        # --- Fuel energy over the year (MMBtu, total) ---
        self.fuel_consumed = self.mysum((self.process_heat_requirement.to('MMBtu/hr') / self.thermal_efficiency) * Q_('1 hr')).to('MMBtu')
        
        # --- Fuel spend (USD/yr) ---
        self.year_one_energy_cost = (self.gas_price * self.fuel_consumed) / Q_('1 yr')
        
        # --- Emissions (ton/yr) and cost (USD/yr) ---
        self.year_one_emissions = (self.fuel_consumed * self.emissions_factor) / Q_('1 yr')
        self.year_one_cost_of_emissions = (self.emissions_carbon_price * self.year_one_emissions).to('USD/yr')
        
        # --- Sum (USD/yr) ---
        self.year_one_operating_costs = (
            self.year_one_fixed_o_and_m
            + self.year_one_variable_o_and_m
            + self.year_one_energy_cost
            + self.year_one_cost_of_emissions
        ).to('USD/yr')
        
        # --- LCOH ---
        self.LCOH = self.calculate_LCOH(self.capital_cost, self.year_one_operating_costs)
        #break out capex and opex
        self.Levelized_Capex = self.calculate_LCOH(self.capital_cost,Q_(0, 'USD/yr'))
        self.Levelized_OpEx = self.calculate_LCOH(Q_(0, 'USD'),self.year_one_operating_costs)
        

        if self.print_results: 
            print('Capital Cost: {:,~.2fP}'.format(self.capital_cost))
            print('Heat Delivered: {:,~.2fP}'.format(self.mysum(self.process_heat_requirement.to('MMBtu/hr')*Q_('1 hr'))))
            print('Fuel Consumed: {:,~.2fP}'.format(self.fuel_consumed))
            print('Capacity Factor: {:~.3fP}'.format(self.capacity_factor))
            print('One Year Fixed O&M Costs: {:,~.2fP}'.format(self.year_one_fixed_o_and_m))
            print('One Year Variable O&M Costs: {:,~.2fP}'.format(self.year_one_variable_o_and_m))
            print('One Year Energy Costs: {:,~.2fP}'.format(self.year_one_energy_cost))
            print('One Year Operating Costs: {:,~.2fP}'.format(self.year_one_operating_costs))
            print('Overall LCOH: {:,~.2fP}'.format(self.LCOH.to('USD/MMBtu')))
            print('Levelized Capex: {:,~.2fP}'.format(self.Levelized_Capex.to('USD/MMBtu')))
            print('Levelized Opex: {:,~.2fP}'.format(self.Levelized_OpEx.to('USD/MMBtu')))
            print('Year 1 Emissions: {:,~.2fP}'.format(self.year_one_emissions))

    def write_output(self, filename):
        data = [
            ['Cold Temperature Available', '{:~.2fP}'.format(self.cold_temperature)],
            ['Hot Temperature Desired', '{:~.2fP}'.format(self.hot_temperature)],
            ['Gas Heater Efficiency', '{:~.3fP}'.format(self.thermal_efficiency)],
            ['Process Heat Average', '{:~.2fP}'.format(np.mean(self.process_heat_requirement.to('MMBtu/hr')))],
            ['Process Heat Average', '{:~.2fP}'.format(np.mean(self.process_heat_requirement.to('kW')))],
            ['Gas Price', '{:~.2fP}'.format(self.gas_price)],
            ['Capacity Factor', '{:~.3fP}'.format(np.mean(self.capacity_factor))],
            ['Project Lifetime', '{:~.2fP}'.format(self.lifetime)],
            ['Power in Average', '{:~.2fP}'.format(self.average_power_in)],
            ['Annual Energy In', '{:~.2fP}'.format(self.annual_energy_in)],
            ['Fuel Used', '{:~.2fP}'.format(self.fuel_consumed)],
            ['Capital Cost Per Unit', '{:,~.2fP}'.format(self.specific_capital_cost)],
            ['Fixed O&M Costs', '{:,~.2fP}'.format(self.fixed_o_and_m_per_size)],
            ['Variable O&M Costs', '{:,~.2fP}'.format(self.variable_o_and_m)],
            ['Capital Cost', '{:,~.2fP}'.format(self.capital_cost)],
            ['Emissions', '{:~.2fP}'.format(self.year_one_emissions)],
            ['Year One Energy Costs', '{:,~.2fP}'.format(self.year_one_energy_cost)],
            ['Year One Fixed O&M Costs', '{:,~.2fP}'.format(self.year_one_fixed_o_and_m)],
            ['Year One Variable O&M Costs', '{:,~.2fP}'.format(self.year_one_variable_o_and_m)],
            ['Year One Total Operating Costs', '{:,~.2fP}'.format(self.year_one_operating_costs)],
            ['LCOH', '{:,~.2fP}'.format(self.LCOH)]
            ]
        
        df_output = pd.DataFrame(data,columns=['Variable','Value'])
        df_output.to_csv('output/'+filename+'.csv')
        if self.print_results: print('Writing all output to a file')


    def run_all(self,filename):
        self.calculate_costs()
        if self.write_output_file: self.write_output(filename)


