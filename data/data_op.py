import pandas as pd
from data_fun import clean_wdi_file,add_world_row,clean_colum,add_base_to_all,add_to_all,fill_population,world_r,metrics_pc,fill_early_history_energy_pc,interpolate_energy_pc,fill_energy_pc_world_avg,fill_no_energy_pc_countries

co2csv = "clean_data/co2.csv"
econocsv = "clean_data/industry.csv"
energycsv = "clean_data/energy.csv"
forecsv = "clean_data/forest.csv"
fosscsv = "clean_data/fossil.csv"
popucsv = "clean_data/population.csv"
renewcsv = "clean_data/renewable.csv"
allcsv = "clean_data/all_data.csv"

co2 = pd.read_csv(co2csv,encoding="utf-8-sig",on_bad_lines="skip")
econo = pd.read_csv(econocsv,encoding="utf-8-sig",on_bad_lines="skip")
energy = pd.read_csv(energycsv,encoding="utf-8-sig",on_bad_lines="skip")
fore = pd.read_csv(forecsv,encoding="utf-8-sig",on_bad_lines="skip")
foss = pd.read_csv(fosscsv,encoding="utf-8-sig",on_bad_lines="skip")
popu = pd.read_csv(popucsv,encoding="utf-8-sig",on_bad_lines="skip")
renew = pd.read_csv(renewcsv,encoding="utf-8-sig",on_bad_lines="skip")
all = pd.read_csv(allcsv,encoding="utf-8-sig",on_bad_lines="skip")


#print(co2)
#print(econo)
#print(energy)
#print(fore)
#print(foss)
#print(popu)
#print(renew)
print(all)

#world_r(allcsv)

#add_world_row(co2csv)
#add_world_row(econocsv)
#add_world_row(energycsv)
#add_world_row(forecsv)
#add_world_row(fosscsv)
#add_world_row(popucsv)
#add_world_row(renewcsv)

#add_base_to_all(co2csv,allcsv)

#add_to_all(energycsv,allcsv,"Energy")
#add_to_all(econocsv,allcsv,"Industry")
#add_to_all(forecsv,allcsv,"Forest")
#add_to_all(fosscsv,allcsv,"Fossil")
#add_to_all(popucsv,allcsv,"Population")
#add_to_all(renewcsv,allcsv,"Renewable")

# Apply per country
#all = all.groupby("Country Name", group_keys=False).apply(fill_population)

#all.to_csv(allcsv, index=False)

#metrics_pc(allcsv,["Energy"])
e_pc = "all_data_with_pc.csv"
#fill_early_history_energy_pc(e_pc)

#interpolate_energy_pc(e_pc)

#fill_energy_pc_world_avg(e_pc)

#world_r(e_pc)

#fill_no_energy_pc_countries(e_pc)

add_world_row(
    input_path=e_pc,
    per_capita_cols=["Energy_PC"]
)
