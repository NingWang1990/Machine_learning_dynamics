import sys
sys.path.append("../Code")
import numpy as np
import matplotlib.pyplot as plt
import os
from os import path
from get_pareto import Point, ParetoSet
from RPN_to_pytorch import RPN_to_pytorch
from RPN_to_eq import RPN_to_eq
from S_NN_train import NN_train
from S_NN_eval import NN_eval
from S_symmetry import *
from S_separability import *
from S_change_output import *
from S_brute_force import brute_force
from S_combine_pareto import combine_pareto
from S_get_number_DL import get_number_DL
from sympy.parsing.sympy_parser import parse_expr
from sympy import preorder_traversal, count_ops
from S_polyfit import polyfit
from S_get_symbolic_expr_error import get_symbolic_expr_error
from S_add_snap_expr_on_pareto import add_snap_expr_on_pareto
from S_add_sym_on_pareto import add_sym_on_pareto
from S_run_bf_polyfit import run_bf_polyfit
from S_final_gd import final_gd
from S_add_bf_on_numbers_on_pareto import add_bf_on_numbers_on_pareto
from dimensionalAnalysis import dimensionalAnalysis
from S_add_snap_expr_on_pareto_polyfit import add_snap_expr_on_pareto_polyfit


pathdir = './'
filename = 'phase_field_OneD'
polyfit_deg = 4
test_percentage = 20
DR_file = ""
filename_orig = filename
# Split the data into train and test set 
input_data = np.loadtxt(pathdir+filename)
sep_idx = np.random.permutation(len(input_data))
train_data = input_data[sep_idx[0:(100-test_percentage)*len(input_data)//100]]
test_data = input_data[sep_idx[test_percentage*len(input_data)//100:len(input_data)]]
np.savetxt(pathdir+filename+"_train",train_data)
if test_data.size != 0:
    np.savetxt(pathdir+filename+"_test",test_data)

PA = ParetoSet()

filename_train = filename+"_train"

# run polyfit on the data
print("Checking polyfit \n")
polyfit_result = polyfit(polyfit_deg, filename_train)
eqn = str(polyfit_result[0])

polyfit_err = get_symbolic_expr_error(pathdir,filename,eqn)
expr = parse_expr(eqn)
is_atomic_number = lambda expr: expr.is_Atom and expr.is_number
numbers_expr = [subexpression for subexpression in preorder_traversal(expr) if is_atomic_number(subexpression)]
complexity = 0
for j in numbers_expr:
    complexity = complexity + get_number_DL(float(j))
try:
    # Add the complexity due to symbols
    n_variables = len(polyfit_result[0].free_symbols)
    n_operations = len(count_ops(polyfit_result[0],visual=True).free_symbols)
    if n_operations!=0 or n_variables!=0:
        complexity = complexity + (n_variables+n_operations)*np.log2((n_variables+n_operations))
except:
    pass

#run zero snap on polyfit output
PA_poly = ParetoSet()
PA_poly.add(Point(x=complexity, y=polyfit_err, data=str(eqn)))
PA_poly = add_snap_expr_on_pareto_polyfit(pathdir, filename_train, str(eqn), PA_poly) 
for l in range(len(PA_poly.get_pareto_points())):
    PA.add(Point(PA_poly.get_pareto_points()[l][0],PA_poly.get_pareto_points()[l][1],PA_poly.get_pareto_points()[l][2]))

print("Complexity  RMSE  Expression")
for pareto_i in range(len(PA.get_pareto_points())):
    print(PA.get_pareto_points()[pareto_i])

PA_list = PA.get_pareto_points()

# Run gradient descent on the data one more time                                                                                                                          
for i in range(len(PA_list)):
    try:
        gd_update = final_gd(pathdir+filename,PA_list[i][-1])
        PA.add(Point(x=gd_update[1],y=gd_update[0],data=gd_update[2]))
    except:
        continue
  
PA_list = PA.get_pareto_points()
for j in range(len(PA_list)):
    PA = add_snap_expr_on_pareto_polyfit(pathdir,filename_train,PA_list[j][-1],PA)

list_dt = np.array(PA.get_pareto_points())
data_file_len = len(np.loadtxt(pathdir+filename))
log_err = []
log_err_all = []
for i in range(len(list_dt)):
    log_err = log_err + [np.log2(float(list_dt[i][1]))]
    log_err_all = log_err_all + [data_file_len*np.log2(float(list_dt[i][1]))]
log_err = np.array(log_err)
log_err_all = np.array(log_err_all)

# Try the found expressions on the test data                                                                                                                                  
if DR_file=="" and test_data.size != 0:
    test_errors = []
    for i in range(len(list_dt)):
        test_errors = test_errors + [get_symbolic_expr_error(pathdir,filename+"_test",str(list_dt[i][-1]))]
    test_errors = np.array(test_errors)
    # Save all the data to file                                                                                                                                               
    save_data = np.column_stack((test_errors,log_err,log_err_all,list_dt))
else:
    save_data = np.column_stack((log_err,log_err_all,list_dt))
np.savetxt("results/solution_%s" %filename_orig,save_data,fmt="%s")
