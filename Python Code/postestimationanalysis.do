import delimited "D:\GitHub\PaperCienciaDados-PunoPeru\Python Code\post_estimation.csv", clear

preserve

*false negatives
keep if predictions==0

* no se conoce
logit false_negatives inlaid_wall painted_wall paved_track terrain_track paths lighting_pole  independent_house  own_house title_ownership concrete_walls concrete_floor concrete_roof  water_network potable_water water_quality_chlorine water_daily_access  electric_lighting  candle_lighting  other_lighting  electric_cooking    wood_cooking other_cooking manure_cooking  phone cellphone cabletv internet rural altitude overcrowding fullbedroom_house young_adult adult old_adult  woman married literacy no_education basic_education technic_education college_education  illness accident healthy chronic_illness medical_attention contributory_hi subsidized_hi  disabilities employment radio tvcolor tvnoncolor sound_equipment dvd video_recorder computer_laptop electric_iron electric_mixer gas_stove kerosene_stove refrigerator washing_machine microwave_oven sewing_machine bicycle car motorcycle tricycle mototaxi truck gpc
* improvised_house failure
* fuel_lighting failure
* generator_lighting failure
* gas_network_cooking failure
* no_ligthing

*base
*no_public_good
*drainage_network
*charcoal_cooking
*no_cooking
*old
*posgraduate_education
*no_hi

*varianza
*glp_cooking
*household_house



*quedarse con las 10 con mayor Z
margins, dydx(*) atmeans post
marginsplot, xdimension() xline(0) horizontal

coefplot, horizontal keep(chronic_illness radio car motorcycle dvd concrete_roof sound_equipment rural electric_cooking employment ) sort(, descending) xline(0)

restore

preserve
*false negatives
keep if predictions==1

* no se conoce
logit false_positive inlaid_wall painted_wall paved_track terrain_track paths lighting_pole  independent_house  own_house title_ownership concrete_walls concrete_floor concrete_roof  water_network potable_water  water_daily_access  electric_lighting  candle_lighting  other_lighting  wood_cooking other_cooking manure_cooking   cellphone cabletv internet rural altitude overcrowding fullbedroom_house young_adult adult old_adult  woman married literacy no_education basic_education technic_education   illness accident healthy chronic_illness medical_attention contributory_hi subsidized_hi  disabilities employment radio tvcolor tvnoncolor sound_equipment dvd  computer_laptop electric_iron electric_mixer gas_stove  sewing_machine bicycle car motorcycle tricycle mototaxi  gpc

* PERFECT FAILURE
*kerosene_stove
*water_quality_chlorine
*refrigerator
*washing_machine
*microwave_oven
*truck

*colinear
*electric_cooking
*phone
*college_education
*video_recorder

margins, dydx(*) atmeans post
marginsplot, xdimension() xline(0) horizontal

coefplot, horizontal keep(woman tricycle married) sort(, descending) xline(0)

restore
