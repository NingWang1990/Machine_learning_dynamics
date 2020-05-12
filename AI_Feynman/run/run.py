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
polyfit_deg = 1
test_percentage = 20

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
print ("====================starting running zero snap========================================")
PA_poly = add_snap_expr_on_pareto_polyfit(pathdir, filename_train, str(eqn), PA_poly) 

for l in range(len(PA_poly.get_pareto_points())):
    PA.add(Point(PA_poly.get_pareto_points()[l][0],PA_poly.get_pareto_points()[l][1],PA_poly.get_pareto_points()[l][2]))

print("Complexity  RMSE  Expression")
for pareto_i in range(len(PA.get_pareto_points())):
    print(PA.get_pareto_points()[pareto_i])
