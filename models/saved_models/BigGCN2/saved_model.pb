��
��
B
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
t

SegmentMax	
data"T
segment_ids"Tindices
output"T"
Ttype:
2	"
Tindicestype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
SparseTensorDenseMatMul
	a_indices"Tindices
a_values"T
a_shape	
b"T
product"T"	
Ttype"
Tindicestype0	:
2	"
	adjoint_abool( "
	adjoint_bbool( 
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
�
UnsortedSegmentSum	
data"T
segment_ids"Tindices
num_segments"Tnumsegments
output"T" 
Ttype:
2	"
Tindicestype:
2	" 
Tnumsegmentstype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.02v2.4.0-2-g5485ec964e78��	
�
model/ecc_conv/root_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*+
shared_namemodel/ecc_conv/root_kernel
�
.model/ecc_conv/root_kernel/Read/ReadVariableOpReadVariableOpmodel/ecc_conv/root_kernel*
_output_shapes

:@*
dtype0
�
model/gcn_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_namemodel/gcn_conv/kernel

)model/gcn_conv/kernel/Read/ReadVariableOpReadVariableOpmodel/gcn_conv/kernel*
_output_shapes

:@@*
dtype0
�
model/gcn_conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*(
shared_namemodel/gcn_conv_1/kernel
�
+model/gcn_conv_1/kernel/Read/ReadVariableOpReadVariableOpmodel/gcn_conv_1/kernel*
_output_shapes
:	@�*
dtype0
�
model/gcn_conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*(
shared_namemodel/gcn_conv_2/kernel
�
+model/gcn_conv_2/kernel/Read/ReadVariableOpReadVariableOpmodel/gcn_conv_2/kernel* 
_output_shapes
:
��*
dtype0
�
model/gcn_conv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*(
shared_namemodel/gcn_conv_3/kernel
�
+model/gcn_conv_3/kernel/Read/ReadVariableOpReadVariableOpmodel/gcn_conv_3/kernel* 
_output_shapes
:
��*
dtype0
�
model/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*%
shared_namemodel/dense_6/kernel
}
(model/dense_6/kernel/Read/ReadVariableOpReadVariableOpmodel/dense_6/kernel*
_output_shapes

:@*
dtype0
|
model/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namemodel/dense_6/bias
u
&model/dense_6/bias/Read/ReadVariableOpReadVariableOpmodel/dense_6/bias*
_output_shapes
:*
dtype0
�
model/ecc_conv/FGN_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *,
shared_namemodel/ecc_conv/FGN_0/kernel
�
/model/ecc_conv/FGN_0/kernel/Read/ReadVariableOpReadVariableOpmodel/ecc_conv/FGN_0/kernel*
_output_shapes

: *
dtype0
�
model/ecc_conv/FGN_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*,
shared_namemodel/ecc_conv/FGN_1/kernel
�
/model/ecc_conv/FGN_1/kernel/Read/ReadVariableOpReadVariableOpmodel/ecc_conv/FGN_1/kernel*
_output_shapes

: @*
dtype0
�
model/ecc_conv/FGN_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*,
shared_namemodel/ecc_conv/FGN_2/kernel
�
/model/ecc_conv/FGN_2/kernel/Read/ReadVariableOpReadVariableOpmodel/ecc_conv/FGN_2/kernel*
_output_shapes

:@@*
dtype0
�
model/ecc_conv/FGN_out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*.
shared_namemodel/ecc_conv/FGN_out/kernel
�
1model/ecc_conv/FGN_out/kernel/Read/ReadVariableOpReadVariableOpmodel/ecc_conv/FGN_out/kernel*
_output_shapes
:	@�*
dtype0
�
model/ecc_conv/FGN_out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namemodel/ecc_conv/FGN_out/bias
�
/model/ecc_conv/FGN_out/bias/Read/ReadVariableOpReadVariableOpmodel/ecc_conv/FGN_out/bias*
_output_shapes	
:�*
dtype0
�
model/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*#
shared_namemodel/dense/kernel
{
&model/dense/kernel/Read/ReadVariableOpReadVariableOpmodel/dense/kernel* 
_output_shapes
:
��*
dtype0
y
model/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*!
shared_namemodel/dense/bias
r
$model/dense/bias/Read/ReadVariableOpReadVariableOpmodel/dense/bias*
_output_shapes	
:�*
dtype0
�
model/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_namemodel/dense_1/kernel

(model/dense_1/kernel/Read/ReadVariableOpReadVariableOpmodel/dense_1/kernel* 
_output_shapes
:
��*
dtype0
}
model/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_namemodel/dense_1/bias
v
&model/dense_1/bias/Read/ReadVariableOpReadVariableOpmodel/dense_1/bias*
_output_shapes	
:�*
dtype0
�
model/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_namemodel/dense_2/kernel

(model/dense_2/kernel/Read/ReadVariableOpReadVariableOpmodel/dense_2/kernel* 
_output_shapes
:
��*
dtype0
}
model/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_namemodel/dense_2/bias
v
&model/dense_2/bias/Read/ReadVariableOpReadVariableOpmodel/dense_2/bias*
_output_shapes	
:�*
dtype0
�
model/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*%
shared_namemodel/dense_3/kernel
~
(model/dense_3/kernel/Read/ReadVariableOpReadVariableOpmodel/dense_3/kernel*
_output_shapes
:	�@*
dtype0
|
model/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_namemodel/dense_3/bias
u
&model/dense_3/bias/Read/ReadVariableOpReadVariableOpmodel/dense_3/bias*
_output_shapes
:@*
dtype0
�
model/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*%
shared_namemodel/dense_4/kernel
}
(model/dense_4/kernel/Read/ReadVariableOpReadVariableOpmodel/dense_4/kernel*
_output_shapes

:@@*
dtype0
|
model/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_namemodel/dense_4/bias
u
&model/dense_4/bias/Read/ReadVariableOpReadVariableOpmodel/dense_4/bias*
_output_shapes
:@*
dtype0
�
model/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*%
shared_namemodel/dense_5/kernel
}
(model/dense_5/kernel/Read/ReadVariableOpReadVariableOpmodel/dense_5/kernel*
_output_shapes

:@@*
dtype0
|
model/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_namemodel/dense_5/bias
u
&model/dense_5/bias/Read/ReadVariableOpReadVariableOpmodel/dense_5/bias*
_output_shapes
:@*
dtype0

NoOpNoOp
�I
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�I
value�IB�I B�I
�
ECC1
GCN1
GCN2
GCN3
GCN4
Pool

decode
d2
		optimizer

loss

signatures
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
�
kwargs_keys
kernel_network
kernel_network_layers
root_kernel
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
�
kwargs_keys

kernel
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
 	keras_api
�
!kwargs_keys

"kernel
##_self_saveable_object_factories
$	variables
%regularization_losses
&trainable_variables
'	keras_api
�
(kwargs_keys

)kernel
#*_self_saveable_object_factories
+	variables
,regularization_losses
-trainable_variables
.	keras_api
�
/kwargs_keys

0kernel
#1_self_saveable_object_factories
2	variables
3regularization_losses
4trainable_variables
5	keras_api
w
#6_self_saveable_object_factories
7	variables
8regularization_losses
9trainable_variables
:	keras_api
*
;0
<1
=2
>3
?4
@5
�

Akernel
Bbias
#C_self_saveable_object_factories
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
 
 
 
 
�
0
H1
I2
J3
K4
L5
6
"7
)8
09
M10
N11
O12
P13
Q14
R15
S16
T17
U18
V19
W20
X21
A22
B23
 
�
0
H1
I2
J3
K4
L5
6
"7
)8
09
M10
N11
O12
P13
Q14
R15
S16
T17
U18
V19
W20
X21
A22
B23
�

Ylayers
Zlayer_regularization_losses
	variables
regularization_losses
[metrics
\non_trainable_variables
trainable_variables
]layer_metrics
 
 

^0
_1
`2
a3
[Y
VARIABLE_VALUEmodel/ecc_conv/root_kernel+ECC1/root_kernel/.ATTRIBUTES/VARIABLE_VALUE
 
*
0
H1
I2
J3
K4
L5
 
*
0
H1
I2
J3
K4
L5
�

blayers
clayer_regularization_losses
	variables
regularization_losses
dmetrics
enon_trainable_variables
trainable_variables
flayer_metrics
 
QO
VARIABLE_VALUEmodel/gcn_conv/kernel&GCN1/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

0
 

0
�

glayers
hlayer_regularization_losses
	variables
regularization_losses
imetrics
jnon_trainable_variables
trainable_variables
klayer_metrics
 
SQ
VARIABLE_VALUEmodel/gcn_conv_1/kernel&GCN2/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

"0
 

"0
�

llayers
mlayer_regularization_losses
$	variables
%regularization_losses
nmetrics
onon_trainable_variables
&trainable_variables
player_metrics
 
SQ
VARIABLE_VALUEmodel/gcn_conv_2/kernel&GCN3/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

)0
 

)0
�

qlayers
rlayer_regularization_losses
+	variables
,regularization_losses
smetrics
tnon_trainable_variables
-trainable_variables
ulayer_metrics
 
SQ
VARIABLE_VALUEmodel/gcn_conv_3/kernel&GCN4/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

00
 

00
�

vlayers
wlayer_regularization_losses
2	variables
3regularization_losses
xmetrics
ynon_trainable_variables
4trainable_variables
zlayer_metrics
 
 
 
 
�

{layers
|layer_regularization_losses
7	variables
8regularization_losses
}metrics
~non_trainable_variables
9trainable_variables
layer_metrics
�

Mkernel
Nbias
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
�

Okernel
Pbias
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
�

Qkernel
Rbias
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
�

Skernel
Tbias
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
�

Ukernel
Vbias
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
�

Wkernel
Xbias
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
NL
VARIABLE_VALUEmodel/dense_6/kernel$d2/kernel/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEmodel/dense_6/bias"d2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

A0
B1
 

A0
B1
�
�layers
 �layer_regularization_losses
D	variables
Eregularization_losses
�metrics
�non_trainable_variables
Ftrainable_variables
�layer_metrics
WU
VARIABLE_VALUEmodel/ecc_conv/FGN_0/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEmodel/ecc_conv/FGN_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEmodel/ecc_conv/FGN_2/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEmodel/ecc_conv/FGN_out/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEmodel/ecc_conv/FGN_out/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEmodel/dense/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEmodel/dense/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEmodel/dense_1/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEmodel/dense_1/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEmodel/dense_2/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEmodel/dense_2/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEmodel/dense_3/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEmodel/dense_3/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEmodel/dense_4/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEmodel/dense_4/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEmodel/dense_5/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEmodel/dense_5/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
^
0
1
2
3
4
5
;6
<7
=8
>9
?10
@11
12
 
 
 
 
�

Hkernel
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
�

Ikernel
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
�

Jkernel
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
�

Kkernel
Lbias
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api

^0
_1
`2
a3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

M0
N1
 

M0
N1
�
�layers
 �layer_regularization_losses
�	variables
�regularization_losses
�metrics
�non_trainable_variables
�trainable_variables
�layer_metrics
 

O0
P1
 

O0
P1
�
�layers
 �layer_regularization_losses
�	variables
�regularization_losses
�metrics
�non_trainable_variables
�trainable_variables
�layer_metrics
 

Q0
R1
 

Q0
R1
�
�layers
 �layer_regularization_losses
�	variables
�regularization_losses
�metrics
�non_trainable_variables
�trainable_variables
�layer_metrics
 

S0
T1
 

S0
T1
�
�layers
 �layer_regularization_losses
�	variables
�regularization_losses
�metrics
�non_trainable_variables
�trainable_variables
�layer_metrics
 

U0
V1
 

U0
V1
�
�layers
 �layer_regularization_losses
�	variables
�regularization_losses
�metrics
�non_trainable_variables
�trainable_variables
�layer_metrics
 

W0
X1
 

W0
X1
�
�layers
 �layer_regularization_losses
�	variables
�regularization_losses
�metrics
�non_trainable_variables
�trainable_variables
�layer_metrics
 
 
 
 
 
 

H0
 

H0
�
�layers
 �layer_regularization_losses
�	variables
�regularization_losses
�metrics
�non_trainable_variables
�trainable_variables
�layer_metrics
 

I0
 

I0
�
�layers
 �layer_regularization_losses
�	variables
�regularization_losses
�metrics
�non_trainable_variables
�trainable_variables
�layer_metrics
 

J0
 

J0
�
�layers
 �layer_regularization_losses
�	variables
�regularization_losses
�metrics
�non_trainable_variables
�trainable_variables
�layer_metrics
 

K0
L1
 

K0
L1
�
�layers
 �layer_regularization_losses
�	variables
�regularization_losses
�metrics
�non_trainable_variables
�trainable_variables
�layer_metrics
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
y
serving_default_args_0Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
{
serving_default_args_0_1Placeholder*'
_output_shapes
:���������*
dtype0	*
shape:���������
s
serving_default_args_0_2Placeholder*#
_output_shapes
:���������*
dtype0*
shape:���������
a
serving_default_args_0_3Placeholder*
_output_shapes
:*
dtype0	*
shape:
{
serving_default_args_0_4Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
s
serving_default_args_0_5Placeholder*#
_output_shapes
:���������*
dtype0	*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_0_1serving_default_args_0_2serving_default_args_0_3serving_default_args_0_4serving_default_args_0_5model/ecc_conv/FGN_0/kernelmodel/ecc_conv/FGN_1/kernelmodel/ecc_conv/FGN_2/kernelmodel/ecc_conv/FGN_out/kernelmodel/ecc_conv/FGN_out/biasmodel/ecc_conv/root_kernelmodel/gcn_conv/kernelmodel/gcn_conv_1/kernelmodel/gcn_conv_2/kernelmodel/gcn_conv_3/kernelmodel/dense/kernelmodel/dense/biasmodel/dense_1/kernelmodel/dense_1/biasmodel/dense_2/kernelmodel/dense_2/biasmodel/dense_3/kernelmodel/dense_3/biasmodel/dense_4/kernelmodel/dense_4/biasmodel/dense_5/kernelmodel/dense_5/biasmodel/dense_6/kernelmodel/dense_6/bias*)
Tin"
 2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference_signature_wrapper_40522
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename.model/ecc_conv/root_kernel/Read/ReadVariableOp)model/gcn_conv/kernel/Read/ReadVariableOp+model/gcn_conv_1/kernel/Read/ReadVariableOp+model/gcn_conv_2/kernel/Read/ReadVariableOp+model/gcn_conv_3/kernel/Read/ReadVariableOp(model/dense_6/kernel/Read/ReadVariableOp&model/dense_6/bias/Read/ReadVariableOp/model/ecc_conv/FGN_0/kernel/Read/ReadVariableOp/model/ecc_conv/FGN_1/kernel/Read/ReadVariableOp/model/ecc_conv/FGN_2/kernel/Read/ReadVariableOp1model/ecc_conv/FGN_out/kernel/Read/ReadVariableOp/model/ecc_conv/FGN_out/bias/Read/ReadVariableOp&model/dense/kernel/Read/ReadVariableOp$model/dense/bias/Read/ReadVariableOp(model/dense_1/kernel/Read/ReadVariableOp&model/dense_1/bias/Read/ReadVariableOp(model/dense_2/kernel/Read/ReadVariableOp&model/dense_2/bias/Read/ReadVariableOp(model/dense_3/kernel/Read/ReadVariableOp&model/dense_3/bias/Read/ReadVariableOp(model/dense_4/kernel/Read/ReadVariableOp&model/dense_4/bias/Read/ReadVariableOp(model/dense_5/kernel/Read/ReadVariableOp&model/dense_5/bias/Read/ReadVariableOpConst*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference__traced_save_40622
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemodel/ecc_conv/root_kernelmodel/gcn_conv/kernelmodel/gcn_conv_1/kernelmodel/gcn_conv_2/kernelmodel/gcn_conv_3/kernelmodel/dense_6/kernelmodel/dense_6/biasmodel/ecc_conv/FGN_0/kernelmodel/ecc_conv/FGN_1/kernelmodel/ecc_conv/FGN_2/kernelmodel/ecc_conv/FGN_out/kernelmodel/ecc_conv/FGN_out/biasmodel/dense/kernelmodel/dense/biasmodel/dense_1/kernelmodel/dense_1/biasmodel/dense_2/kernelmodel/dense_2/biasmodel/dense_3/kernelmodel/dense_3/biasmodel/dense_4/kernelmodel/dense_4/biasmodel/dense_5/kernelmodel/dense_5/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_restore_40704׸
�	
�
*__inference_gcn_conv_3_layer_call_fn_37190
inputs_0

inputs	
inputs_1
inputs_2	
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2unknown*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_gcn_conv_3_layer_call_and_return_conditional_losses_371812
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*S
_input_shapesB
@:����������:���������:���������::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�	
�
B__inference_dense_3_layer_call_and_return_conditional_losses_36905

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�d
�
!__inference__traced_restore_40704
file_prefix/
+assignvariableop_model_ecc_conv_root_kernel,
(assignvariableop_1_model_gcn_conv_kernel.
*assignvariableop_2_model_gcn_conv_1_kernel.
*assignvariableop_3_model_gcn_conv_2_kernel.
*assignvariableop_4_model_gcn_conv_3_kernel+
'assignvariableop_5_model_dense_6_kernel)
%assignvariableop_6_model_dense_6_bias2
.assignvariableop_7_model_ecc_conv_fgn_0_kernel2
.assignvariableop_8_model_ecc_conv_fgn_1_kernel2
.assignvariableop_9_model_ecc_conv_fgn_2_kernel5
1assignvariableop_10_model_ecc_conv_fgn_out_kernel3
/assignvariableop_11_model_ecc_conv_fgn_out_bias*
&assignvariableop_12_model_dense_kernel(
$assignvariableop_13_model_dense_bias,
(assignvariableop_14_model_dense_1_kernel*
&assignvariableop_15_model_dense_1_bias,
(assignvariableop_16_model_dense_2_kernel*
&assignvariableop_17_model_dense_2_bias,
(assignvariableop_18_model_dense_3_kernel*
&assignvariableop_19_model_dense_3_bias,
(assignvariableop_20_model_dense_4_kernel*
&assignvariableop_21_model_dense_4_bias,
(assignvariableop_22_model_dense_5_kernel*
&assignvariableop_23_model_dense_5_bias
identity_25��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B+ECC1/root_kernel/.ATTRIBUTES/VARIABLE_VALUEB&GCN1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&GCN2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&GCN3/kernel/.ATTRIBUTES/VARIABLE_VALUEB&GCN4/kernel/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp+assignvariableop_model_ecc_conv_root_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp(assignvariableop_1_model_gcn_conv_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp*assignvariableop_2_model_gcn_conv_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp*assignvariableop_3_model_gcn_conv_2_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp*assignvariableop_4_model_gcn_conv_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp'assignvariableop_5_model_dense_6_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp%assignvariableop_6_model_dense_6_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp.assignvariableop_7_model_ecc_conv_fgn_0_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp.assignvariableop_8_model_ecc_conv_fgn_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp.assignvariableop_9_model_ecc_conv_fgn_2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp1assignvariableop_10_model_ecc_conv_fgn_out_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp/assignvariableop_11_model_ecc_conv_fgn_out_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_model_dense_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp$assignvariableop_13_model_dense_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp(assignvariableop_14_model_dense_1_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp&assignvariableop_15_model_dense_1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp(assignvariableop_16_model_dense_2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp&assignvariableop_17_model_dense_2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp(assignvariableop_18_model_dense_3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp&assignvariableop_19_model_dense_3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp(assignvariableop_20_model_dense_4_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp&assignvariableop_21_model_dense_4_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp(assignvariableop_22_model_dense_5_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp&assignvariableop_23_model_dense_5_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_239
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_24�
Identity_25IdentityIdentity_24:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_25"#
identity_25Identity_25:output:0*u
_input_shapesd
b: ::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�7
�
__inference__traced_save_40622
file_prefix9
5savev2_model_ecc_conv_root_kernel_read_readvariableop4
0savev2_model_gcn_conv_kernel_read_readvariableop6
2savev2_model_gcn_conv_1_kernel_read_readvariableop6
2savev2_model_gcn_conv_2_kernel_read_readvariableop6
2savev2_model_gcn_conv_3_kernel_read_readvariableop3
/savev2_model_dense_6_kernel_read_readvariableop1
-savev2_model_dense_6_bias_read_readvariableop:
6savev2_model_ecc_conv_fgn_0_kernel_read_readvariableop:
6savev2_model_ecc_conv_fgn_1_kernel_read_readvariableop:
6savev2_model_ecc_conv_fgn_2_kernel_read_readvariableop<
8savev2_model_ecc_conv_fgn_out_kernel_read_readvariableop:
6savev2_model_ecc_conv_fgn_out_bias_read_readvariableop1
-savev2_model_dense_kernel_read_readvariableop/
+savev2_model_dense_bias_read_readvariableop3
/savev2_model_dense_1_kernel_read_readvariableop1
-savev2_model_dense_1_bias_read_readvariableop3
/savev2_model_dense_2_kernel_read_readvariableop1
-savev2_model_dense_2_bias_read_readvariableop3
/savev2_model_dense_3_kernel_read_readvariableop1
-savev2_model_dense_3_bias_read_readvariableop3
/savev2_model_dense_4_kernel_read_readvariableop1
-savev2_model_dense_4_bias_read_readvariableop3
/savev2_model_dense_5_kernel_read_readvariableop1
-savev2_model_dense_5_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B+ECC1/root_kernel/.ATTRIBUTES/VARIABLE_VALUEB&GCN1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&GCN2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&GCN3/kernel/.ATTRIBUTES/VARIABLE_VALUEB&GCN4/kernel/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:05savev2_model_ecc_conv_root_kernel_read_readvariableop0savev2_model_gcn_conv_kernel_read_readvariableop2savev2_model_gcn_conv_1_kernel_read_readvariableop2savev2_model_gcn_conv_2_kernel_read_readvariableop2savev2_model_gcn_conv_3_kernel_read_readvariableop/savev2_model_dense_6_kernel_read_readvariableop-savev2_model_dense_6_bias_read_readvariableop6savev2_model_ecc_conv_fgn_0_kernel_read_readvariableop6savev2_model_ecc_conv_fgn_1_kernel_read_readvariableop6savev2_model_ecc_conv_fgn_2_kernel_read_readvariableop8savev2_model_ecc_conv_fgn_out_kernel_read_readvariableop6savev2_model_ecc_conv_fgn_out_bias_read_readvariableop-savev2_model_dense_kernel_read_readvariableop+savev2_model_dense_bias_read_readvariableop/savev2_model_dense_1_kernel_read_readvariableop-savev2_model_dense_1_bias_read_readvariableop/savev2_model_dense_2_kernel_read_readvariableop-savev2_model_dense_2_bias_read_readvariableop/savev2_model_dense_3_kernel_read_readvariableop-savev2_model_dense_3_bias_read_readvariableop/savev2_model_dense_4_kernel_read_readvariableop-savev2_model_dense_4_bias_read_readvariableop/savev2_model_dense_5_kernel_read_readvariableop-savev2_model_dense_5_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :@:@@:	@�:
��:
��:@:: : @:@@:	@�:�:
��:�:
��:�:
��:�:	�@:@:@@:@:@@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@:$ 

_output_shapes

:@@:%!

_output_shapes
:	@�:&"
 
_output_shapes
:
��:&"
 
_output_shapes
:
��:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

: :$	 

_output_shapes

: @:$
 

_output_shapes

:@@:%!

_output_shapes
:	@�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:

_output_shapes
: 
�	
�
B__inference_dense_4_layer_call_and_return_conditional_losses_37200

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�J
�
@__inference_model_layer_call_and_return_conditional_losses_37412

inputs
inputs_1	
inputs_2
inputs_3	
inputs_4
inputs_5	
ecc_conv_217778
ecc_conv_217780
ecc_conv_217782
ecc_conv_217784
ecc_conv_217786
ecc_conv_217788
gcn_conv_217791
gcn_conv_1_217794
gcn_conv_2_217797
gcn_conv_3_217800
dense_217804
dense_217806
dense_1_217810
dense_1_217812
dense_2_217816
dense_2_217818
dense_3_217822
dense_3_217824
dense_4_217828
dense_4_217830
dense_5_217834
dense_5_217836
dense_6_217840
dense_6_217842
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall� ecc_conv/StatefulPartitionedCall� gcn_conv/StatefulPartitionedCall�"gcn_conv_1/StatefulPartitionedCall�"gcn_conv_2/StatefulPartitionedCall�"gcn_conv_3/StatefulPartitionedCall�
 ecc_conv/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4ecc_conv_217778ecc_conv_217780ecc_conv_217782ecc_conv_217784ecc_conv_217786ecc_conv_217788*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_ecc_conv_layer_call_and_return_conditional_losses_371542"
 ecc_conv/StatefulPartitionedCall�
 gcn_conv/StatefulPartitionedCallStatefulPartitionedCall)ecc_conv/StatefulPartitionedCall:output:0inputs_1inputs_2inputs_3gcn_conv_217791*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gcn_conv_layer_call_and_return_conditional_losses_370752"
 gcn_conv/StatefulPartitionedCall�
"gcn_conv_1/StatefulPartitionedCallStatefulPartitionedCall)gcn_conv/StatefulPartitionedCall:output:0inputs_1inputs_2inputs_3gcn_conv_1_217794*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_gcn_conv_1_layer_call_and_return_conditional_losses_368312$
"gcn_conv_1/StatefulPartitionedCall�
"gcn_conv_2/StatefulPartitionedCallStatefulPartitionedCall+gcn_conv_1/StatefulPartitionedCall:output:0inputs_1inputs_2inputs_3gcn_conv_2_217797*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_gcn_conv_2_layer_call_and_return_conditional_losses_368862$
"gcn_conv_2/StatefulPartitionedCall�
"gcn_conv_3/StatefulPartitionedCallStatefulPartitionedCall+gcn_conv_2/StatefulPartitionedCall:output:0inputs_1inputs_2inputs_3gcn_conv_3_217800*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_gcn_conv_3_layer_call_and_return_conditional_losses_371812$
"gcn_conv_3/StatefulPartitionedCall�
global_max_pool/PartitionedCallPartitionedCall+gcn_conv_3/StatefulPartitionedCall:output:0inputs_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_global_max_pool_layer_call_and_return_conditional_losses_370902!
global_max_pool/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall(global_max_pool/PartitionedCall:output:0dense_217804dense_217806*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_369152
dense/StatefulPartitionedCallo
TanhTanh&dense/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:����������2
Tanh�
dense_1/StatefulPartitionedCallStatefulPartitionedCallTanh:y:0dense_1_217810dense_1_217812*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_373602!
dense_1/StatefulPartitionedCallu
Tanh_1Tanh(dense_1/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:����������2
Tanh_1�
dense_2/StatefulPartitionedCallStatefulPartitionedCall
Tanh_1:y:0dense_2_217816dense_2_217818*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_369252!
dense_2/StatefulPartitionedCallu
Tanh_2Tanh(dense_2/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:����������2
Tanh_2�
dense_3/StatefulPartitionedCallStatefulPartitionedCall
Tanh_2:y:0dense_3_217822dense_3_217824*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_369052!
dense_3/StatefulPartitionedCallt
Tanh_3Tanh(dense_3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������@2
Tanh_3�
dense_4/StatefulPartitionedCallStatefulPartitionedCall
Tanh_3:y:0dense_4_217828dense_4_217830*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_372002!
dense_4/StatefulPartitionedCallt
Tanh_4Tanh(dense_4/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������@2
Tanh_4�
dense_5/StatefulPartitionedCallStatefulPartitionedCall
Tanh_4:y:0dense_5_217834dense_5_217836*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_373502!
dense_5/StatefulPartitionedCallt
Tanh_5Tanh(dense_5/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������@2
Tanh_5�
dense_6/StatefulPartitionedCallStatefulPartitionedCall
Tanh_5:y:0dense_6_217840dense_6_217842*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_368502!
dense_6/StatefulPartitionedCall�
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall!^ecc_conv/StatefulPartitionedCall!^gcn_conv/StatefulPartitionedCall#^gcn_conv_1/StatefulPartitionedCall#^gcn_conv_2/StatefulPartitionedCall#^gcn_conv_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������:���������:���������::���������:���������::::::::::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2D
 ecc_conv/StatefulPartitionedCall ecc_conv/StatefulPartitionedCall2D
 gcn_conv/StatefulPartitionedCall gcn_conv/StatefulPartitionedCall2H
"gcn_conv_1/StatefulPartitionedCall"gcn_conv_1/StatefulPartitionedCall2H
"gcn_conv_2/StatefulPartitionedCall"gcn_conv_2/StatefulPartitionedCall2H
"gcn_conv_3/StatefulPartitionedCall"gcn_conv_3/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
*__inference_gcn_conv_1_layer_call_fn_36840
inputs_0

inputs	
inputs_1
inputs_2	
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2unknown*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_gcn_conv_1_layer_call_and_return_conditional_losses_368312
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:���������@:���������:���������::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�J
�
@__inference_model_layer_call_and_return_conditional_losses_37498

inputs
inputs_1	
inputs_2
inputs_3	
inputs_4
inputs_5	
ecc_conv_217912
ecc_conv_217914
ecc_conv_217916
ecc_conv_217918
ecc_conv_217920
ecc_conv_217922
gcn_conv_217925
gcn_conv_1_217928
gcn_conv_2_217931
gcn_conv_3_217934
dense_217938
dense_217940
dense_1_217944
dense_1_217946
dense_2_217950
dense_2_217952
dense_3_217956
dense_3_217958
dense_4_217962
dense_4_217964
dense_5_217968
dense_5_217970
dense_6_217974
dense_6_217976
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall� ecc_conv/StatefulPartitionedCall� gcn_conv/StatefulPartitionedCall�"gcn_conv_1/StatefulPartitionedCall�"gcn_conv_2/StatefulPartitionedCall�"gcn_conv_3/StatefulPartitionedCall�
 ecc_conv/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4ecc_conv_217912ecc_conv_217914ecc_conv_217916ecc_conv_217918ecc_conv_217920ecc_conv_217922*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_ecc_conv_layer_call_and_return_conditional_losses_371542"
 ecc_conv/StatefulPartitionedCall�
 gcn_conv/StatefulPartitionedCallStatefulPartitionedCall)ecc_conv/StatefulPartitionedCall:output:0inputs_1inputs_2inputs_3gcn_conv_217925*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gcn_conv_layer_call_and_return_conditional_losses_370752"
 gcn_conv/StatefulPartitionedCall�
"gcn_conv_1/StatefulPartitionedCallStatefulPartitionedCall)gcn_conv/StatefulPartitionedCall:output:0inputs_1inputs_2inputs_3gcn_conv_1_217928*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_gcn_conv_1_layer_call_and_return_conditional_losses_368312$
"gcn_conv_1/StatefulPartitionedCall�
"gcn_conv_2/StatefulPartitionedCallStatefulPartitionedCall+gcn_conv_1/StatefulPartitionedCall:output:0inputs_1inputs_2inputs_3gcn_conv_2_217931*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_gcn_conv_2_layer_call_and_return_conditional_losses_368862$
"gcn_conv_2/StatefulPartitionedCall�
"gcn_conv_3/StatefulPartitionedCallStatefulPartitionedCall+gcn_conv_2/StatefulPartitionedCall:output:0inputs_1inputs_2inputs_3gcn_conv_3_217934*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_gcn_conv_3_layer_call_and_return_conditional_losses_371812$
"gcn_conv_3/StatefulPartitionedCall�
global_max_pool/PartitionedCallPartitionedCall+gcn_conv_3/StatefulPartitionedCall:output:0inputs_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_global_max_pool_layer_call_and_return_conditional_losses_370902!
global_max_pool/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall(global_max_pool/PartitionedCall:output:0dense_217938dense_217940*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_369152
dense/StatefulPartitionedCallo
TanhTanh&dense/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:����������2
Tanh�
dense_1/StatefulPartitionedCallStatefulPartitionedCallTanh:y:0dense_1_217944dense_1_217946*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_373602!
dense_1/StatefulPartitionedCallu
Tanh_1Tanh(dense_1/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:����������2
Tanh_1�
dense_2/StatefulPartitionedCallStatefulPartitionedCall
Tanh_1:y:0dense_2_217950dense_2_217952*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_369252!
dense_2/StatefulPartitionedCallu
Tanh_2Tanh(dense_2/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:����������2
Tanh_2�
dense_3/StatefulPartitionedCallStatefulPartitionedCall
Tanh_2:y:0dense_3_217956dense_3_217958*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_369052!
dense_3/StatefulPartitionedCallt
Tanh_3Tanh(dense_3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������@2
Tanh_3�
dense_4/StatefulPartitionedCallStatefulPartitionedCall
Tanh_3:y:0dense_4_217962dense_4_217964*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_372002!
dense_4/StatefulPartitionedCallt
Tanh_4Tanh(dense_4/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������@2
Tanh_4�
dense_5/StatefulPartitionedCallStatefulPartitionedCall
Tanh_4:y:0dense_5_217968dense_5_217970*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_373502!
dense_5/StatefulPartitionedCallt
Tanh_5Tanh(dense_5/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������@2
Tanh_5�
dense_6/StatefulPartitionedCallStatefulPartitionedCall
Tanh_5:y:0dense_6_217974dense_6_217976*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_368502!
dense_6/StatefulPartitionedCall�
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall!^ecc_conv/StatefulPartitionedCall!^gcn_conv/StatefulPartitionedCall#^gcn_conv_1/StatefulPartitionedCall#^gcn_conv_2/StatefulPartitionedCall#^gcn_conv_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������:���������:���������::���������:���������::::::::::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2D
 ecc_conv/StatefulPartitionedCall ecc_conv/StatefulPartitionedCall2D
 gcn_conv/StatefulPartitionedCall gcn_conv/StatefulPartitionedCall2H
"gcn_conv_1/StatefulPartitionedCall"gcn_conv_1/StatefulPartitionedCall2H
"gcn_conv_2/StatefulPartitionedCall"gcn_conv_2/StatefulPartitionedCall2H
"gcn_conv_3/StatefulPartitionedCall"gcn_conv_3/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs
�B
�
C__inference_ecc_conv_layer_call_and_return_conditional_losses_37154

inputs
inputs_1	
inputs_2
inputs_3	
inputs_4(
$fgn_0_matmul_readvariableop_resource(
$fgn_1_matmul_readvariableop_resource(
$fgn_2_matmul_readvariableop_resource*
&fgn_out_matmul_readvariableop_resource+
'fgn_out_biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity��FGN_0/MatMul/ReadVariableOp�FGN_1/MatMul/ReadVariableOp�FGN_2/MatMul/ReadVariableOp�FGN_out/BiasAdd/ReadVariableOp�FGN_out/MatMul/ReadVariableOp�MatMul_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shape}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice/stack�
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice�
FGN_0/MatMul/ReadVariableOpReadVariableOp$fgn_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
FGN_0/MatMul/ReadVariableOp�
FGN_0/MatMulMatMulinputs_4#FGN_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
FGN_0/MatMulj

FGN_0/ReluReluFGN_0/MatMul:product:0*
T0*'
_output_shapes
:��������� 2

FGN_0/Relu�
FGN_1/MatMul/ReadVariableOpReadVariableOp$fgn_1_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02
FGN_1/MatMul/ReadVariableOp�
FGN_1/MatMulMatMulFGN_0/Relu:activations:0#FGN_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
FGN_1/MatMulj

FGN_1/ReluReluFGN_1/MatMul:product:0*
T0*'
_output_shapes
:���������@2

FGN_1/Relu�
FGN_2/MatMul/ReadVariableOpReadVariableOp$fgn_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
FGN_2/MatMul/ReadVariableOp�
FGN_2/MatMulMatMulFGN_1/Relu:activations:0#FGN_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
FGN_2/MatMulj

FGN_2/ReluReluFGN_2/MatMul:product:0*
T0*'
_output_shapes
:���������@2

FGN_2/Relu�
FGN_out/MatMul/ReadVariableOpReadVariableOp&fgn_out_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02
FGN_out/MatMul/ReadVariableOp�
FGN_out/MatMulMatMulFGN_2/Relu:activations:0%FGN_out/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
FGN_out/MatMul�
FGN_out/BiasAdd/ReadVariableOpReadVariableOp'fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
FGN_out/BiasAdd/ReadVariableOp�
FGN_out/BiasAddBiasAddFGN_out/MatMul:product:0&FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
FGN_out/BiasAdds
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   @   2
Reshape/shape�
ReshapeReshapeFGN_out/BiasAdd:output:0Reshape/shape:output:0*
T0*+
_output_shapes
:���������@2	
Reshape
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ����2
strided_slice_1/stack�
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ����2
strided_slice_1/stack_1�
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2�
strided_slice_1StridedSliceinputs_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ����2
strided_slice_2/stack�
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1�
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2�
strided_slice_2StridedSliceinputs_1strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis�
GatherV2GatherV2inputsstrided_slice_2:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:���������2

GatherV2�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_3/stack�
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_3/stack_1�
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_3/stack_2�
strided_slice_3StridedSliceGatherV2:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask2
strided_slice_3�
MatMulBatchMatMulV2strided_slice_3:output:0Reshape:output:0*
T0*+
_output_shapes
:���������@2
MatMul�
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_4/stack�
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_4/stack_1�
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_4/stack_2�
strided_slice_4StridedSliceMatMul:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4�
UnsortedSegmentSumUnsortedSegmentSumstrided_slice_4:output:0strided_slice_1:output:0strided_slice:output:0*
T0*
Tindices0	*'
_output_shapes
:���������@2
UnsortedSegmentSum�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulinputsMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2

MatMul_1v
addAddV2UnsortedSegmentSum:output:0MatMul_1:product:0*
T0*'
_output_shapes
:���������@2
addO
ReluReluadd:z:0*
T0*'
_output_shapes
:���������@2
Relu�
IdentityIdentityRelu:activations:0^FGN_0/MatMul/ReadVariableOp^FGN_1/MatMul/ReadVariableOp^FGN_2/MatMul/ReadVariableOp^FGN_out/BiasAdd/ReadVariableOp^FGN_out/MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:���������:���������:���������::���������::::::2:
FGN_0/MatMul/ReadVariableOpFGN_0/MatMul/ReadVariableOp2:
FGN_1/MatMul/ReadVariableOpFGN_1/MatMul/ReadVariableOp2:
FGN_2/MatMul/ReadVariableOpFGN_2/MatMul/ReadVariableOp2@
FGN_out/BiasAdd/ReadVariableOpFGN_out/BiasAdd/ReadVariableOp2>
FGN_out/MatMul/ReadVariableOpFGN_out/MatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
B__inference_dense_2_layer_call_and_return_conditional_losses_36925

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_37532
inputs_0

inputs	
inputs_1
inputs_2	

inputs_2_0
inputs_3	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2
inputs_2_0inputs_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*)
Tin"
 2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_374982
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������:���������:���������::���������:���������::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2:MI
#
_output_shapes
:���������
"
_user_specified_name
inputs/3
�
�
%__inference_model_layer_call_fn_37446
inputs_0

inputs	
inputs_1
inputs_2	

inputs_2_0
inputs_3	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2
inputs_2_0inputs_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*)
Tin"
 2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_374122
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������:���������:���������::���������:���������::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2:MI
#
_output_shapes
:���������
"
_user_specified_name
inputs/3
�
�
C__inference_gcn_conv_layer_call_and_return_conditional_losses_37075

inputs
inputs_1	
inputs_2
inputs_3	"
matmul_readvariableop_resource
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1inputs_2inputs_3MatMul:product:0*
T0*'
_output_shapes
:���������@21
/SparseTensorDenseMatMul/SparseTensorDenseMatMul�
ReluRelu9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*'
_output_shapes
:���������@2
Relu~
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:���������@:���������:���������::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�
v
J__inference_global_max_pool_layer_call_and_return_conditional_losses_36536
inputs_0
inputs_1	
identity}

SegmentMax
SegmentMaxinputs_0inputs_1*
T0*
Tindices0	*(
_output_shapes
:����������2

SegmentMaxh
IdentityIdentitySegmentMax:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:����������:���������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:���������
"
_user_specified_name
inputs/1
�	
�
B__inference_dense_5_layer_call_and_return_conditional_losses_37350

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
E__inference_gcn_conv_1_layer_call_and_return_conditional_losses_37340
inputs_0

inputs	
inputs_1
inputs_2	"
matmul_readvariableop_resource
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul/ReadVariableOpv
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2MatMul:product:0*
T0*(
_output_shapes
:����������21
/SparseTensorDenseMatMul/SparseTensorDenseMatMul�
ReluRelu9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:����������2
Relu
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:���������@:���������:���������::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�
�
(__inference_ecc_conv_layer_call_fn_37169
inputs_0

inputs	
inputs_1
inputs_2	

inputs_2_0
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2
inputs_2_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_ecc_conv_layer_call_and_return_conditional_losses_371542
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:���������:���������:���������::���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2
�
�
C__inference_gcn_conv_layer_call_and_return_conditional_losses_36874
inputs_0

inputs	
inputs_1
inputs_2	"
matmul_readvariableop_resource
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2MatMul:product:0*
T0*'
_output_shapes
:���������@21
/SparseTensorDenseMatMul/SparseTensorDenseMatMul�
ReluRelu9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*'
_output_shapes
:���������@2
Relu~
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:���������@:���������:���������::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�	
�
B__inference_dense_6_layer_call_and_return_conditional_losses_36850

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
E__inference_gcn_conv_2_layer_call_and_return_conditional_losses_36886

inputs
inputs_1	
inputs_2
inputs_3	"
matmul_readvariableop_resource
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1inputs_2inputs_3MatMul:product:0*
T0*(
_output_shapes
:����������21
/SparseTensorDenseMatMul/SparseTensorDenseMatMul�
ReluRelu9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:����������2
Relu
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*S
_input_shapesB
@:����������:���������:���������::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
��
�
@__inference_model_layer_call_and_return_conditional_losses_37328
inputs_0

inputs	
inputs_1
inputs_2	

inputs_2_0
inputs_3	1
-ecc_conv_fgn_0_matmul_readvariableop_resource1
-ecc_conv_fgn_1_matmul_readvariableop_resource1
-ecc_conv_fgn_2_matmul_readvariableop_resource3
/ecc_conv_fgn_out_matmul_readvariableop_resource4
0ecc_conv_fgn_out_biasadd_readvariableop_resource-
)ecc_conv_matmul_1_readvariableop_resource+
'gcn_conv_matmul_readvariableop_resource-
)gcn_conv_1_matmul_readvariableop_resource-
)gcn_conv_2_matmul_readvariableop_resource-
)gcn_conv_3_matmul_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp�$ecc_conv/FGN_0/MatMul/ReadVariableOp�$ecc_conv/FGN_1/MatMul/ReadVariableOp�$ecc_conv/FGN_2/MatMul/ReadVariableOp�'ecc_conv/FGN_out/BiasAdd/ReadVariableOp�&ecc_conv/FGN_out/MatMul/ReadVariableOp� ecc_conv/MatMul_1/ReadVariableOp�gcn_conv/MatMul/ReadVariableOp� gcn_conv_1/MatMul/ReadVariableOp� gcn_conv_2/MatMul/ReadVariableOp� gcn_conv_3/MatMul/ReadVariableOpX
ecc_conv/ShapeShapeinputs_0*
T0*
_output_shapes
:2
ecc_conv/Shape�
ecc_conv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
ecc_conv/strided_slice/stack�
ecc_conv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2 
ecc_conv/strided_slice/stack_1�
ecc_conv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
ecc_conv/strided_slice/stack_2�
ecc_conv/strided_sliceStridedSliceecc_conv/Shape:output:0%ecc_conv/strided_slice/stack:output:0'ecc_conv/strided_slice/stack_1:output:0'ecc_conv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ecc_conv/strided_slice�
$ecc_conv/FGN_0/MatMul/ReadVariableOpReadVariableOp-ecc_conv_fgn_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype02&
$ecc_conv/FGN_0/MatMul/ReadVariableOp�
ecc_conv/FGN_0/MatMulMatMul
inputs_2_0,ecc_conv/FGN_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
ecc_conv/FGN_0/MatMul�
ecc_conv/FGN_0/ReluReluecc_conv/FGN_0/MatMul:product:0*
T0*'
_output_shapes
:��������� 2
ecc_conv/FGN_0/Relu�
$ecc_conv/FGN_1/MatMul/ReadVariableOpReadVariableOp-ecc_conv_fgn_1_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02&
$ecc_conv/FGN_1/MatMul/ReadVariableOp�
ecc_conv/FGN_1/MatMulMatMul!ecc_conv/FGN_0/Relu:activations:0,ecc_conv/FGN_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
ecc_conv/FGN_1/MatMul�
ecc_conv/FGN_1/ReluReluecc_conv/FGN_1/MatMul:product:0*
T0*'
_output_shapes
:���������@2
ecc_conv/FGN_1/Relu�
$ecc_conv/FGN_2/MatMul/ReadVariableOpReadVariableOp-ecc_conv_fgn_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02&
$ecc_conv/FGN_2/MatMul/ReadVariableOp�
ecc_conv/FGN_2/MatMulMatMul!ecc_conv/FGN_1/Relu:activations:0,ecc_conv/FGN_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
ecc_conv/FGN_2/MatMul�
ecc_conv/FGN_2/ReluReluecc_conv/FGN_2/MatMul:product:0*
T0*'
_output_shapes
:���������@2
ecc_conv/FGN_2/Relu�
&ecc_conv/FGN_out/MatMul/ReadVariableOpReadVariableOp/ecc_conv_fgn_out_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02(
&ecc_conv/FGN_out/MatMul/ReadVariableOp�
ecc_conv/FGN_out/MatMulMatMul!ecc_conv/FGN_2/Relu:activations:0.ecc_conv/FGN_out/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
ecc_conv/FGN_out/MatMul�
'ecc_conv/FGN_out/BiasAdd/ReadVariableOpReadVariableOp0ecc_conv_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'ecc_conv/FGN_out/BiasAdd/ReadVariableOp�
ecc_conv/FGN_out/BiasAddBiasAdd!ecc_conv/FGN_out/MatMul:product:0/ecc_conv/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
ecc_conv/FGN_out/BiasAdd�
ecc_conv/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   @   2
ecc_conv/Reshape/shape�
ecc_conv/ReshapeReshape!ecc_conv/FGN_out/BiasAdd:output:0ecc_conv/Reshape/shape:output:0*
T0*+
_output_shapes
:���������@2
ecc_conv/Reshape�
ecc_conv/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ����2 
ecc_conv/strided_slice_1/stack�
 ecc_conv/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ����2"
 ecc_conv/strided_slice_1/stack_1�
 ecc_conv/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 ecc_conv/strided_slice_1/stack_2�
ecc_conv/strided_slice_1StridedSliceinputs'ecc_conv/strided_slice_1/stack:output:0)ecc_conv/strided_slice_1/stack_1:output:0)ecc_conv/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
ecc_conv/strided_slice_1�
ecc_conv/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ����2 
ecc_conv/strided_slice_2/stack�
 ecc_conv/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 ecc_conv/strided_slice_2/stack_1�
 ecc_conv/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 ecc_conv/strided_slice_2/stack_2�
ecc_conv/strided_slice_2StridedSliceinputs'ecc_conv/strided_slice_2/stack:output:0)ecc_conv/strided_slice_2/stack_1:output:0)ecc_conv/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
ecc_conv/strided_slice_2r
ecc_conv/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
ecc_conv/GatherV2/axis�
ecc_conv/GatherV2GatherV2inputs_0!ecc_conv/strided_slice_2:output:0ecc_conv/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:���������2
ecc_conv/GatherV2�
ecc_conv/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2 
ecc_conv/strided_slice_3/stack�
 ecc_conv/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2"
 ecc_conv/strided_slice_3/stack_1�
 ecc_conv/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2"
 ecc_conv/strided_slice_3/stack_2�
ecc_conv/strided_slice_3StridedSliceecc_conv/GatherV2:output:0'ecc_conv/strided_slice_3/stack:output:0)ecc_conv/strided_slice_3/stack_1:output:0)ecc_conv/strided_slice_3/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask2
ecc_conv/strided_slice_3�
ecc_conv/MatMulBatchMatMulV2!ecc_conv/strided_slice_3:output:0ecc_conv/Reshape:output:0*
T0*+
_output_shapes
:���������@2
ecc_conv/MatMul�
ecc_conv/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2 
ecc_conv/strided_slice_4/stack�
 ecc_conv/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2"
 ecc_conv/strided_slice_4/stack_1�
 ecc_conv/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2"
 ecc_conv/strided_slice_4/stack_2�
ecc_conv/strided_slice_4StridedSliceecc_conv/MatMul:output:0'ecc_conv/strided_slice_4/stack:output:0)ecc_conv/strided_slice_4/stack_1:output:0)ecc_conv/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*

begin_mask*
end_mask*
shrink_axis_mask2
ecc_conv/strided_slice_4�
ecc_conv/UnsortedSegmentSumUnsortedSegmentSum!ecc_conv/strided_slice_4:output:0!ecc_conv/strided_slice_1:output:0ecc_conv/strided_slice:output:0*
T0*
Tindices0	*'
_output_shapes
:���������@2
ecc_conv/UnsortedSegmentSum�
 ecc_conv/MatMul_1/ReadVariableOpReadVariableOp)ecc_conv_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02"
 ecc_conv/MatMul_1/ReadVariableOp�
ecc_conv/MatMul_1MatMulinputs_0(ecc_conv/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
ecc_conv/MatMul_1�
ecc_conv/addAddV2$ecc_conv/UnsortedSegmentSum:output:0ecc_conv/MatMul_1:product:0*
T0*'
_output_shapes
:���������@2
ecc_conv/addj
ecc_conv/ReluReluecc_conv/add:z:0*
T0*'
_output_shapes
:���������@2
ecc_conv/Relu�
gcn_conv/MatMul/ReadVariableOpReadVariableOp'gcn_conv_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
gcn_conv/MatMul/ReadVariableOp�
gcn_conv/MatMulMatMulecc_conv/Relu:activations:0&gcn_conv/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
gcn_conv/MatMul�
8gcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2gcn_conv/MatMul:product:0*
T0*'
_output_shapes
:���������@2:
8gcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMul�
gcn_conv/ReluReluBgcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*'
_output_shapes
:���������@2
gcn_conv/Relu�
 gcn_conv_1/MatMul/ReadVariableOpReadVariableOp)gcn_conv_1_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02"
 gcn_conv_1/MatMul/ReadVariableOp�
gcn_conv_1/MatMulMatMulgcn_conv/Relu:activations:0(gcn_conv_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gcn_conv_1/MatMul�
:gcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2gcn_conv_1/MatMul:product:0*
T0*(
_output_shapes
:����������2<
:gcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMul�
gcn_conv_1/ReluReluDgcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:����������2
gcn_conv_1/Relu�
 gcn_conv_2/MatMul/ReadVariableOpReadVariableOp)gcn_conv_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02"
 gcn_conv_2/MatMul/ReadVariableOp�
gcn_conv_2/MatMulMatMulgcn_conv_1/Relu:activations:0(gcn_conv_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gcn_conv_2/MatMul�
:gcn_conv_2/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2gcn_conv_2/MatMul:product:0*
T0*(
_output_shapes
:����������2<
:gcn_conv_2/SparseTensorDenseMatMul/SparseTensorDenseMatMul�
gcn_conv_2/ReluReluDgcn_conv_2/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:����������2
gcn_conv_2/Relu�
 gcn_conv_3/MatMul/ReadVariableOpReadVariableOp)gcn_conv_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02"
 gcn_conv_3/MatMul/ReadVariableOp�
gcn_conv_3/MatMulMatMulgcn_conv_2/Relu:activations:0(gcn_conv_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gcn_conv_3/MatMul�
:gcn_conv_3/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2gcn_conv_3/MatMul:product:0*
T0*(
_output_shapes
:����������2<
:gcn_conv_3/SparseTensorDenseMatMul/SparseTensorDenseMatMul�
gcn_conv_3/ReluReluDgcn_conv_3/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:����������2
gcn_conv_3/Relu�
global_max_pool/SegmentMax
SegmentMaxgcn_conv_3/Relu:activations:0inputs_3*
T0*
Tindices0	*(
_output_shapes
:����������2
global_max_pool/SegmentMax�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMul#global_max_pool/SegmentMax:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAdd_
TanhTanhdense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
Tanh�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulTanh:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/BiasAdde
Tanh_1Tanhdense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
Tanh_1�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMul
Tanh_1:y:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_2/BiasAdde
Tanh_2Tanhdense_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
Tanh_2�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMul
Tanh_2:y:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_3/MatMul�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_3/BiasAddd
Tanh_3Tanhdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Tanh_3�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_4/MatMul/ReadVariableOp�
dense_4/MatMulMatMul
Tanh_3:y:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_4/MatMul�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_4/BiasAdd/ReadVariableOp�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_4/BiasAddd
Tanh_4Tanhdense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Tanh_4�
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_5/MatMul/ReadVariableOp�
dense_5/MatMulMatMul
Tanh_4:y:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_5/MatMul�
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_5/BiasAdd/ReadVariableOp�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_5/BiasAddd
Tanh_5Tanhdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Tanh_5�
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_6/MatMul/ReadVariableOp�
dense_6/MatMulMatMul
Tanh_5:y:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_6/MatMul�
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp�
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_6/BiasAdd�
IdentityIdentitydense_6/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp%^ecc_conv/FGN_0/MatMul/ReadVariableOp%^ecc_conv/FGN_1/MatMul/ReadVariableOp%^ecc_conv/FGN_2/MatMul/ReadVariableOp(^ecc_conv/FGN_out/BiasAdd/ReadVariableOp'^ecc_conv/FGN_out/MatMul/ReadVariableOp!^ecc_conv/MatMul_1/ReadVariableOp^gcn_conv/MatMul/ReadVariableOp!^gcn_conv_1/MatMul/ReadVariableOp!^gcn_conv_2/MatMul/ReadVariableOp!^gcn_conv_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������:���������:���������::���������:���������::::::::::::::::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2L
$ecc_conv/FGN_0/MatMul/ReadVariableOp$ecc_conv/FGN_0/MatMul/ReadVariableOp2L
$ecc_conv/FGN_1/MatMul/ReadVariableOp$ecc_conv/FGN_1/MatMul/ReadVariableOp2L
$ecc_conv/FGN_2/MatMul/ReadVariableOp$ecc_conv/FGN_2/MatMul/ReadVariableOp2R
'ecc_conv/FGN_out/BiasAdd/ReadVariableOp'ecc_conv/FGN_out/BiasAdd/ReadVariableOp2P
&ecc_conv/FGN_out/MatMul/ReadVariableOp&ecc_conv/FGN_out/MatMul/ReadVariableOp2D
 ecc_conv/MatMul_1/ReadVariableOp ecc_conv/MatMul_1/ReadVariableOp2@
gcn_conv/MatMul/ReadVariableOpgcn_conv/MatMul/ReadVariableOp2D
 gcn_conv_1/MatMul/ReadVariableOp gcn_conv_1/MatMul/ReadVariableOp2D
 gcn_conv_2/MatMul/ReadVariableOp gcn_conv_2/MatMul/ReadVariableOp2D
 gcn_conv_3/MatMul/ReadVariableOp gcn_conv_3/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2:MI
#
_output_shapes
:���������
"
_user_specified_name
inputs/3
�
�
E__inference_gcn_conv_3_layer_call_and_return_conditional_losses_37181

inputs
inputs_1	
inputs_2
inputs_3	"
matmul_readvariableop_resource
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1inputs_2inputs_3MatMul:product:0*
T0*(
_output_shapes
:����������21
/SparseTensorDenseMatMul/SparseTensorDenseMatMul�
ReluRelu9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:����������2
Relu
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*S
_input_shapesB
@:����������:���������:���������::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�
�
E__inference_gcn_conv_1_layer_call_and_return_conditional_losses_36831

inputs
inputs_1	
inputs_2
inputs_3	"
matmul_readvariableop_resource
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1inputs_2inputs_3MatMul:product:0*
T0*(
_output_shapes
:����������21
/SparseTensorDenseMatMul/SparseTensorDenseMatMul�
ReluRelu9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:����������2
Relu
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:���������@:���������:���������::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�
�
 __inference__wrapped_model_40462

args_0
args_0_1	
args_0_2
args_0_3	
args_0_4
args_0_5	
model_40412
model_40414
model_40416
model_40418
model_40420
model_40422
model_40424
model_40426
model_40428
model_40430
model_40432
model_40434
model_40436
model_40438
model_40440
model_40442
model_40444
model_40446
model_40448
model_40450
model_40452
model_40454
model_40456
model_40458
identity��model/StatefulPartitionedCall�
model/StatefulPartitionedCallStatefulPartitionedCallargs_0args_0_1args_0_2args_0_3args_0_4args_0_5model_40412model_40414model_40416model_40418model_40420model_40422model_40424model_40426model_40428model_40430model_40432model_40434model_40436model_40438model_40440model_40442model_40444model_40446model_40448model_40450model_40452model_40454model_40456model_40458*)
Tin"
 2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *1
f,R*
(__inference_restored_function_body_404112
model/StatefulPartitionedCall�
IdentityIdentity&model/StatefulPartitionedCall:output:0^model/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������:���������:���������::���������:���������::::::::::::::::::::::::2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameargs_0:OK
'
_output_shapes
:���������
 
_user_specified_nameargs_0:KG
#
_output_shapes
:���������
 
_user_specified_nameargs_0:B>

_output_shapes
:
 
_user_specified_nameargs_0:OK
'
_output_shapes
:���������
 
_user_specified_nameargs_0:KG
#
_output_shapes
:���������
 
_user_specified_nameargs_0
�
�
E__inference_gcn_conv_2_layer_call_and_return_conditional_losses_36589
inputs_0

inputs	
inputs_1
inputs_2	"
matmul_readvariableop_resource
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpv
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2MatMul:product:0*
T0*(
_output_shapes
:����������21
/SparseTensorDenseMatMul/SparseTensorDenseMatMul�
ReluRelu9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:����������2
Relu
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*S
_input_shapesB
@:����������:���������:���������::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�
�
(__inference_restored_function_body_40411

inputs
inputs_1	
inputs_2
inputs_3	
inputs_4
inputs_5	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*)
Tin"
 2			*
Tout
2*'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_367172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������:���������:���������::���������:���������::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_gcn_conv_3_layer_call_and_return_conditional_losses_36862
inputs_0

inputs	
inputs_1
inputs_2	"
matmul_readvariableop_resource
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpv
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2MatMul:product:0*
T0*(
_output_shapes
:����������21
/SparseTensorDenseMatMul/SparseTensorDenseMatMul�
ReluRelu9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:����������2
Relu
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*S
_input_shapesB
@:����������:���������:���������::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�	
�
(__inference_gcn_conv_layer_call_fn_37084
inputs_0

inputs	
inputs_1
inputs_2	
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2unknown*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gcn_conv_layer_call_and_return_conditional_losses_370752
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:���������@:���������:���������::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�	
�
@__inference_dense_layer_call_and_return_conditional_losses_36915

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
@__inference_model_layer_call_and_return_conditional_losses_36717
inputs_0

inputs	
inputs_1
inputs_2	

inputs_2_0
inputs_3	1
-ecc_conv_fgn_0_matmul_readvariableop_resource1
-ecc_conv_fgn_1_matmul_readvariableop_resource1
-ecc_conv_fgn_2_matmul_readvariableop_resource3
/ecc_conv_fgn_out_matmul_readvariableop_resource4
0ecc_conv_fgn_out_biasadd_readvariableop_resource-
)ecc_conv_matmul_1_readvariableop_resource+
'gcn_conv_matmul_readvariableop_resource-
)gcn_conv_1_matmul_readvariableop_resource-
)gcn_conv_2_matmul_readvariableop_resource-
)gcn_conv_3_matmul_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp�$ecc_conv/FGN_0/MatMul/ReadVariableOp�$ecc_conv/FGN_1/MatMul/ReadVariableOp�$ecc_conv/FGN_2/MatMul/ReadVariableOp�'ecc_conv/FGN_out/BiasAdd/ReadVariableOp�&ecc_conv/FGN_out/MatMul/ReadVariableOp� ecc_conv/MatMul_1/ReadVariableOp�gcn_conv/MatMul/ReadVariableOp� gcn_conv_1/MatMul/ReadVariableOp� gcn_conv_2/MatMul/ReadVariableOp� gcn_conv_3/MatMul/ReadVariableOpX
ecc_conv/ShapeShapeinputs_0*
T0*
_output_shapes
:2
ecc_conv/Shape�
ecc_conv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
ecc_conv/strided_slice/stack�
ecc_conv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2 
ecc_conv/strided_slice/stack_1�
ecc_conv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
ecc_conv/strided_slice/stack_2�
ecc_conv/strided_sliceStridedSliceecc_conv/Shape:output:0%ecc_conv/strided_slice/stack:output:0'ecc_conv/strided_slice/stack_1:output:0'ecc_conv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ecc_conv/strided_slice�
$ecc_conv/FGN_0/MatMul/ReadVariableOpReadVariableOp-ecc_conv_fgn_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype02&
$ecc_conv/FGN_0/MatMul/ReadVariableOp�
ecc_conv/FGN_0/MatMulMatMul
inputs_2_0,ecc_conv/FGN_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
ecc_conv/FGN_0/MatMul�
ecc_conv/FGN_0/ReluReluecc_conv/FGN_0/MatMul:product:0*
T0*'
_output_shapes
:��������� 2
ecc_conv/FGN_0/Relu�
$ecc_conv/FGN_1/MatMul/ReadVariableOpReadVariableOp-ecc_conv_fgn_1_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02&
$ecc_conv/FGN_1/MatMul/ReadVariableOp�
ecc_conv/FGN_1/MatMulMatMul!ecc_conv/FGN_0/Relu:activations:0,ecc_conv/FGN_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
ecc_conv/FGN_1/MatMul�
ecc_conv/FGN_1/ReluReluecc_conv/FGN_1/MatMul:product:0*
T0*'
_output_shapes
:���������@2
ecc_conv/FGN_1/Relu�
$ecc_conv/FGN_2/MatMul/ReadVariableOpReadVariableOp-ecc_conv_fgn_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02&
$ecc_conv/FGN_2/MatMul/ReadVariableOp�
ecc_conv/FGN_2/MatMulMatMul!ecc_conv/FGN_1/Relu:activations:0,ecc_conv/FGN_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
ecc_conv/FGN_2/MatMul�
ecc_conv/FGN_2/ReluReluecc_conv/FGN_2/MatMul:product:0*
T0*'
_output_shapes
:���������@2
ecc_conv/FGN_2/Relu�
&ecc_conv/FGN_out/MatMul/ReadVariableOpReadVariableOp/ecc_conv_fgn_out_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02(
&ecc_conv/FGN_out/MatMul/ReadVariableOp�
ecc_conv/FGN_out/MatMulMatMul!ecc_conv/FGN_2/Relu:activations:0.ecc_conv/FGN_out/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
ecc_conv/FGN_out/MatMul�
'ecc_conv/FGN_out/BiasAdd/ReadVariableOpReadVariableOp0ecc_conv_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'ecc_conv/FGN_out/BiasAdd/ReadVariableOp�
ecc_conv/FGN_out/BiasAddBiasAdd!ecc_conv/FGN_out/MatMul:product:0/ecc_conv/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
ecc_conv/FGN_out/BiasAdd�
ecc_conv/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   @   2
ecc_conv/Reshape/shape�
ecc_conv/ReshapeReshape!ecc_conv/FGN_out/BiasAdd:output:0ecc_conv/Reshape/shape:output:0*
T0*+
_output_shapes
:���������@2
ecc_conv/Reshape�
ecc_conv/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ����2 
ecc_conv/strided_slice_1/stack�
 ecc_conv/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ����2"
 ecc_conv/strided_slice_1/stack_1�
 ecc_conv/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 ecc_conv/strided_slice_1/stack_2�
ecc_conv/strided_slice_1StridedSliceinputs'ecc_conv/strided_slice_1/stack:output:0)ecc_conv/strided_slice_1/stack_1:output:0)ecc_conv/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
ecc_conv/strided_slice_1�
ecc_conv/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ����2 
ecc_conv/strided_slice_2/stack�
 ecc_conv/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 ecc_conv/strided_slice_2/stack_1�
 ecc_conv/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 ecc_conv/strided_slice_2/stack_2�
ecc_conv/strided_slice_2StridedSliceinputs'ecc_conv/strided_slice_2/stack:output:0)ecc_conv/strided_slice_2/stack_1:output:0)ecc_conv/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
ecc_conv/strided_slice_2r
ecc_conv/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
ecc_conv/GatherV2/axis�
ecc_conv/GatherV2GatherV2inputs_0!ecc_conv/strided_slice_2:output:0ecc_conv/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:���������2
ecc_conv/GatherV2�
ecc_conv/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2 
ecc_conv/strided_slice_3/stack�
 ecc_conv/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2"
 ecc_conv/strided_slice_3/stack_1�
 ecc_conv/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2"
 ecc_conv/strided_slice_3/stack_2�
ecc_conv/strided_slice_3StridedSliceecc_conv/GatherV2:output:0'ecc_conv/strided_slice_3/stack:output:0)ecc_conv/strided_slice_3/stack_1:output:0)ecc_conv/strided_slice_3/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask2
ecc_conv/strided_slice_3�
ecc_conv/MatMulBatchMatMulV2!ecc_conv/strided_slice_3:output:0ecc_conv/Reshape:output:0*
T0*+
_output_shapes
:���������@2
ecc_conv/MatMul�
ecc_conv/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2 
ecc_conv/strided_slice_4/stack�
 ecc_conv/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2"
 ecc_conv/strided_slice_4/stack_1�
 ecc_conv/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2"
 ecc_conv/strided_slice_4/stack_2�
ecc_conv/strided_slice_4StridedSliceecc_conv/MatMul:output:0'ecc_conv/strided_slice_4/stack:output:0)ecc_conv/strided_slice_4/stack_1:output:0)ecc_conv/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*

begin_mask*
end_mask*
shrink_axis_mask2
ecc_conv/strided_slice_4�
ecc_conv/UnsortedSegmentSumUnsortedSegmentSum!ecc_conv/strided_slice_4:output:0!ecc_conv/strided_slice_1:output:0ecc_conv/strided_slice:output:0*
T0*
Tindices0	*'
_output_shapes
:���������@2
ecc_conv/UnsortedSegmentSum�
 ecc_conv/MatMul_1/ReadVariableOpReadVariableOp)ecc_conv_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02"
 ecc_conv/MatMul_1/ReadVariableOp�
ecc_conv/MatMul_1MatMulinputs_0(ecc_conv/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
ecc_conv/MatMul_1�
ecc_conv/addAddV2$ecc_conv/UnsortedSegmentSum:output:0ecc_conv/MatMul_1:product:0*
T0*'
_output_shapes
:���������@2
ecc_conv/addj
ecc_conv/ReluReluecc_conv/add:z:0*
T0*'
_output_shapes
:���������@2
ecc_conv/Relu�
gcn_conv/MatMul/ReadVariableOpReadVariableOp'gcn_conv_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
gcn_conv/MatMul/ReadVariableOp�
gcn_conv/MatMulMatMulecc_conv/Relu:activations:0&gcn_conv/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
gcn_conv/MatMul�
8gcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2gcn_conv/MatMul:product:0*
T0*'
_output_shapes
:���������@2:
8gcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMul�
gcn_conv/ReluReluBgcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*'
_output_shapes
:���������@2
gcn_conv/Relu�
 gcn_conv_1/MatMul/ReadVariableOpReadVariableOp)gcn_conv_1_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02"
 gcn_conv_1/MatMul/ReadVariableOp�
gcn_conv_1/MatMulMatMulgcn_conv/Relu:activations:0(gcn_conv_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gcn_conv_1/MatMul�
:gcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2gcn_conv_1/MatMul:product:0*
T0*(
_output_shapes
:����������2<
:gcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMul�
gcn_conv_1/ReluReluDgcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:����������2
gcn_conv_1/Relu�
 gcn_conv_2/MatMul/ReadVariableOpReadVariableOp)gcn_conv_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02"
 gcn_conv_2/MatMul/ReadVariableOp�
gcn_conv_2/MatMulMatMulgcn_conv_1/Relu:activations:0(gcn_conv_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gcn_conv_2/MatMul�
:gcn_conv_2/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2gcn_conv_2/MatMul:product:0*
T0*(
_output_shapes
:����������2<
:gcn_conv_2/SparseTensorDenseMatMul/SparseTensorDenseMatMul�
gcn_conv_2/ReluReluDgcn_conv_2/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:����������2
gcn_conv_2/Relu�
 gcn_conv_3/MatMul/ReadVariableOpReadVariableOp)gcn_conv_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02"
 gcn_conv_3/MatMul/ReadVariableOp�
gcn_conv_3/MatMulMatMulgcn_conv_2/Relu:activations:0(gcn_conv_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gcn_conv_3/MatMul�
:gcn_conv_3/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2gcn_conv_3/MatMul:product:0*
T0*(
_output_shapes
:����������2<
:gcn_conv_3/SparseTensorDenseMatMul/SparseTensorDenseMatMul�
gcn_conv_3/ReluReluDgcn_conv_3/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:����������2
gcn_conv_3/Relu�
global_max_pool/SegmentMax
SegmentMaxgcn_conv_3/Relu:activations:0inputs_3*
T0*
Tindices0	*(
_output_shapes
:����������2
global_max_pool/SegmentMax�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMul#global_max_pool/SegmentMax:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAdd_
TanhTanhdense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
Tanh�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulTanh:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/BiasAdde
Tanh_1Tanhdense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
Tanh_1�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMul
Tanh_1:y:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_2/BiasAdde
Tanh_2Tanhdense_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
Tanh_2�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMul
Tanh_2:y:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_3/MatMul�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_3/BiasAddd
Tanh_3Tanhdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Tanh_3�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_4/MatMul/ReadVariableOp�
dense_4/MatMulMatMul
Tanh_3:y:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_4/MatMul�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_4/BiasAdd/ReadVariableOp�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_4/BiasAddd
Tanh_4Tanhdense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Tanh_4�
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_5/MatMul/ReadVariableOp�
dense_5/MatMulMatMul
Tanh_4:y:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_5/MatMul�
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_5/BiasAdd/ReadVariableOp�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_5/BiasAddd
Tanh_5Tanhdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Tanh_5�
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_6/MatMul/ReadVariableOp�
dense_6/MatMulMatMul
Tanh_5:y:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_6/MatMul�
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp�
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_6/BiasAdd�
IdentityIdentitydense_6/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp%^ecc_conv/FGN_0/MatMul/ReadVariableOp%^ecc_conv/FGN_1/MatMul/ReadVariableOp%^ecc_conv/FGN_2/MatMul/ReadVariableOp(^ecc_conv/FGN_out/BiasAdd/ReadVariableOp'^ecc_conv/FGN_out/MatMul/ReadVariableOp!^ecc_conv/MatMul_1/ReadVariableOp^gcn_conv/MatMul/ReadVariableOp!^gcn_conv_1/MatMul/ReadVariableOp!^gcn_conv_2/MatMul/ReadVariableOp!^gcn_conv_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������:���������:���������::���������:���������::::::::::::::::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2L
$ecc_conv/FGN_0/MatMul/ReadVariableOp$ecc_conv/FGN_0/MatMul/ReadVariableOp2L
$ecc_conv/FGN_1/MatMul/ReadVariableOp$ecc_conv/FGN_1/MatMul/ReadVariableOp2L
$ecc_conv/FGN_2/MatMul/ReadVariableOp$ecc_conv/FGN_2/MatMul/ReadVariableOp2R
'ecc_conv/FGN_out/BiasAdd/ReadVariableOp'ecc_conv/FGN_out/BiasAdd/ReadVariableOp2P
&ecc_conv/FGN_out/MatMul/ReadVariableOp&ecc_conv/FGN_out/MatMul/ReadVariableOp2D
 ecc_conv/MatMul_1/ReadVariableOp ecc_conv/MatMul_1/ReadVariableOp2@
gcn_conv/MatMul/ReadVariableOpgcn_conv/MatMul/ReadVariableOp2D
 gcn_conv_1/MatMul/ReadVariableOp gcn_conv_1/MatMul/ReadVariableOp2D
 gcn_conv_2/MatMul/ReadVariableOp gcn_conv_2/MatMul/ReadVariableOp2D
 gcn_conv_3/MatMul/ReadVariableOp gcn_conv_3/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2:MI
#
_output_shapes
:���������
"
_user_specified_name
inputs/3
�
[
/__inference_global_max_pool_layer_call_fn_37096
inputs_0
inputs_1	
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_global_max_pool_layer_call_and_return_conditional_losses_370902
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:����������:���������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:���������
"
_user_specified_name
inputs/1
�B
�
C__inference_ecc_conv_layer_call_and_return_conditional_losses_37063
inputs_0

inputs	
inputs_1
inputs_2	

inputs_2_0(
$fgn_0_matmul_readvariableop_resource(
$fgn_1_matmul_readvariableop_resource(
$fgn_2_matmul_readvariableop_resource*
&fgn_out_matmul_readvariableop_resource+
'fgn_out_biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity��FGN_0/MatMul/ReadVariableOp�FGN_1/MatMul/ReadVariableOp�FGN_2/MatMul/ReadVariableOp�FGN_out/BiasAdd/ReadVariableOp�FGN_out/MatMul/ReadVariableOp�MatMul_1/ReadVariableOpF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shape}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice/stack�
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice�
FGN_0/MatMul/ReadVariableOpReadVariableOp$fgn_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
FGN_0/MatMul/ReadVariableOp�
FGN_0/MatMulMatMul
inputs_2_0#FGN_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
FGN_0/MatMulj

FGN_0/ReluReluFGN_0/MatMul:product:0*
T0*'
_output_shapes
:��������� 2

FGN_0/Relu�
FGN_1/MatMul/ReadVariableOpReadVariableOp$fgn_1_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02
FGN_1/MatMul/ReadVariableOp�
FGN_1/MatMulMatMulFGN_0/Relu:activations:0#FGN_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
FGN_1/MatMulj

FGN_1/ReluReluFGN_1/MatMul:product:0*
T0*'
_output_shapes
:���������@2

FGN_1/Relu�
FGN_2/MatMul/ReadVariableOpReadVariableOp$fgn_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
FGN_2/MatMul/ReadVariableOp�
FGN_2/MatMulMatMulFGN_1/Relu:activations:0#FGN_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
FGN_2/MatMulj

FGN_2/ReluReluFGN_2/MatMul:product:0*
T0*'
_output_shapes
:���������@2

FGN_2/Relu�
FGN_out/MatMul/ReadVariableOpReadVariableOp&fgn_out_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02
FGN_out/MatMul/ReadVariableOp�
FGN_out/MatMulMatMulFGN_2/Relu:activations:0%FGN_out/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
FGN_out/MatMul�
FGN_out/BiasAdd/ReadVariableOpReadVariableOp'fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
FGN_out/BiasAdd/ReadVariableOp�
FGN_out/BiasAddBiasAddFGN_out/MatMul:product:0&FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
FGN_out/BiasAdds
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   @   2
Reshape/shape�
ReshapeReshapeFGN_out/BiasAdd:output:0Reshape/shape:output:0*
T0*+
_output_shapes
:���������@2	
Reshape
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ����2
strided_slice_1/stack�
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ����2
strided_slice_1/stack_1�
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2�
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ����2
strided_slice_2/stack�
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1�
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2�
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis�
GatherV2GatherV2inputs_0strided_slice_2:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:���������2

GatherV2�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_3/stack�
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_3/stack_1�
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_3/stack_2�
strided_slice_3StridedSliceGatherV2:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask2
strided_slice_3�
MatMulBatchMatMulV2strided_slice_3:output:0Reshape:output:0*
T0*+
_output_shapes
:���������@2
MatMul�
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_4/stack�
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_4/stack_1�
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_4/stack_2�
strided_slice_4StridedSliceMatMul:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4�
UnsortedSegmentSumUnsortedSegmentSumstrided_slice_4:output:0strided_slice_1:output:0strided_slice:output:0*
T0*
Tindices0	*'
_output_shapes
:���������@2
UnsortedSegmentSum�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulinputs_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2

MatMul_1v
addAddV2UnsortedSegmentSum:output:0MatMul_1:product:0*
T0*'
_output_shapes
:���������@2
addO
ReluReluadd:z:0*
T0*'
_output_shapes
:���������@2
Relu�
IdentityIdentityRelu:activations:0^FGN_0/MatMul/ReadVariableOp^FGN_1/MatMul/ReadVariableOp^FGN_2/MatMul/ReadVariableOp^FGN_out/BiasAdd/ReadVariableOp^FGN_out/MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:���������:���������:���������::���������::::::2:
FGN_0/MatMul/ReadVariableOpFGN_0/MatMul/ReadVariableOp2:
FGN_1/MatMul/ReadVariableOpFGN_1/MatMul/ReadVariableOp2:
FGN_2/MatMul/ReadVariableOpFGN_2/MatMul/ReadVariableOp2@
FGN_out/BiasAdd/ReadVariableOpFGN_out/BiasAdd/ReadVariableOp2>
FGN_out/MatMul/ReadVariableOpFGN_out/MatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2
�	
�
B__inference_dense_1_layer_call_and_return_conditional_losses_37360

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_40522

args_0
args_0_1	
args_0_2
args_0_3	
args_0_4
args_0_5	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallargs_0args_0_1args_0_2args_0_3args_0_4args_0_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*)
Tin"
 2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__wrapped_model_404622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������:���������:���������::���������:���������::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameargs_0:QM
'
_output_shapes
:���������
"
_user_specified_name
args_0_1:MI
#
_output_shapes
:���������
"
_user_specified_name
args_0_2:D@

_output_shapes
:
"
_user_specified_name
args_0_3:QM
'
_output_shapes
:���������
"
_user_specified_name
args_0_4:MI
#
_output_shapes
:���������
"
_user_specified_name
args_0_5
�	
�
*__inference_gcn_conv_2_layer_call_fn_36895
inputs_0

inputs	
inputs_1
inputs_2	
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2unknown*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_gcn_conv_2_layer_call_and_return_conditional_losses_368862
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*S
_input_shapesB
@:����������:���������:���������::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�
t
J__inference_global_max_pool_layer_call_and_return_conditional_losses_37090

inputs
inputs_1	
identity{

SegmentMax
SegmentMaxinputsinputs_1*
T0*
Tindices0	*(
_output_shapes
:����������2

SegmentMaxh
IdentityIdentitySegmentMax:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:����������:���������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
9
args_0/
serving_default_args_0:0���������
=
args_0_11
serving_default_args_0_1:0	���������
9
args_0_2-
serving_default_args_0_2:0���������
0
args_0_3$
serving_default_args_0_3:0	
=
args_0_41
serving_default_args_0_4:0���������
9
args_0_5-
serving_default_args_0_5:0	���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
ECC1
GCN1
GCN2
GCN3
GCN4
Pool

decode
d2
		optimizer

loss

signatures
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�_default_save_signature
�__call__"�
_tf_keras_model�{"class_name": "model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "model"}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.001, "decay": 0.0, "rho": 0.9, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
�
kwargs_keys
kernel_network
kernel_network_layers
root_kernel
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "ECCConv", "name": "ecc_conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ecc_conv", "trainable": true, "dtype": "float32", "activation": "relu", "use_bias": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "channels": 64, "kernel_network": [32, 64, 64], "root": true}}
�
kwargs_keys

kernel
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
 	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "GCNConv", "name": "gcn_conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gcn_conv", "trainable": true, "dtype": "float32", "activation": "relu", "use_bias": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "channels": 64}}
�
!kwargs_keys

"kernel
##_self_saveable_object_factories
$	variables
%regularization_losses
&trainable_variables
'	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "GCNConv", "name": "gcn_conv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gcn_conv_1", "trainable": true, "dtype": "float32", "activation": "relu", "use_bias": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "channels": 128}}
�
(kwargs_keys

)kernel
#*_self_saveable_object_factories
+	variables
,regularization_losses
-trainable_variables
.	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "GCNConv", "name": "gcn_conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gcn_conv_2", "trainable": true, "dtype": "float32", "activation": "relu", "use_bias": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "channels": 256}}
�
/kwargs_keys

0kernel
#1_self_saveable_object_factories
2	variables
3regularization_losses
4trainable_variables
5	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "GCNConv", "name": "gcn_conv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gcn_conv_3", "trainable": true, "dtype": "float32", "activation": "relu", "use_bias": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "channels": 512}}
�
#6_self_saveable_object_factories
7	variables
8regularization_losses
9trainable_variables
:	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "GlobalMaxPool", "name": "global_max_pool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_max_pool", "trainable": true, "dtype": "float32"}}
J
;0
<1
=2
>3
?4
@5"
trackable_list_wrapper
�

Akernel
Bbias
#C_self_saveable_object_factories
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
"
	optimizer
 "
trackable_dict_wrapper
-
�serving_default"
signature_map
 "
trackable_dict_wrapper
�
0
H1
I2
J3
K4
L5
6
"7
)8
09
M10
N11
O12
P13
Q14
R15
S16
T17
U18
V19
W20
X21
A22
B23"
trackable_list_wrapper
 "
trackable_list_wrapper
�
0
H1
I2
J3
K4
L5
6
"7
)8
09
M10
N11
O12
P13
Q14
R15
S16
T17
U18
V19
W20
X21
A22
B23"
trackable_list_wrapper
�

Ylayers
Zlayer_regularization_losses
	variables
regularization_losses
[metrics
\non_trainable_variables
trainable_variables
]layer_metrics
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
^0
_1
`2
a3"
trackable_list_wrapper
,:*@2model/ecc_conv/root_kernel
 "
trackable_dict_wrapper
J
0
H1
I2
J3
K4
L5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
H1
I2
J3
K4
L5"
trackable_list_wrapper
�

blayers
clayer_regularization_losses
	variables
regularization_losses
dmetrics
enon_trainable_variables
trainable_variables
flayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%@@2model/gcn_conv/kernel
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
�

glayers
hlayer_regularization_losses
	variables
regularization_losses
imetrics
jnon_trainable_variables
trainable_variables
klayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(	@�2model/gcn_conv_1/kernel
 "
trackable_dict_wrapper
'
"0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
"0"
trackable_list_wrapper
�

llayers
mlayer_regularization_losses
$	variables
%regularization_losses
nmetrics
onon_trainable_variables
&trainable_variables
player_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)
��2model/gcn_conv_2/kernel
 "
trackable_dict_wrapper
'
)0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
)0"
trackable_list_wrapper
�

qlayers
rlayer_regularization_losses
+	variables
,regularization_losses
smetrics
tnon_trainable_variables
-trainable_variables
ulayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)
��2model/gcn_conv_3/kernel
 "
trackable_dict_wrapper
'
00"
trackable_list_wrapper
 "
trackable_list_wrapper
'
00"
trackable_list_wrapper
�

vlayers
wlayer_regularization_losses
2	variables
3regularization_losses
xmetrics
ynon_trainable_variables
4trainable_variables
zlayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

{layers
|layer_regularization_losses
7	variables
8regularization_losses
}metrics
~non_trainable_variables
9trainable_variables
layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

Mkernel
Nbias
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
�

Okernel
Pbias
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
�

Qkernel
Rbias
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
�

Skernel
Tbias
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
�

Ukernel
Vbias
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
�

Wkernel
Xbias
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
&:$@2model/dense_6/kernel
 :2model/dense_6/bias
 "
trackable_dict_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
�
�layers
 �layer_regularization_losses
D	variables
Eregularization_losses
�metrics
�non_trainable_variables
Ftrainable_variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+ 2model/ecc_conv/FGN_0/kernel
-:+ @2model/ecc_conv/FGN_1/kernel
-:+@@2model/ecc_conv/FGN_2/kernel
0:.	@�2model/ecc_conv/FGN_out/kernel
*:(�2model/ecc_conv/FGN_out/bias
&:$
��2model/dense/kernel
:�2model/dense/bias
(:&
��2model/dense_1/kernel
!:�2model/dense_1/bias
(:&
��2model/dense_2/kernel
!:�2model/dense_2/bias
':%	�@2model/dense_3/kernel
 :@2model/dense_3/bias
&:$@@2model/dense_4/kernel
 :@2model/dense_4/bias
&:$@@2model/dense_5/kernel
 :@2model/dense_5/bias
~
0
1
2
3
4
5
;6
<7
=8
>9
?10
@11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�

Hkernel
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "FGN_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "FGN_0", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5]}}
�

Ikernel
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "FGN_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "FGN_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
�

Jkernel
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "FGN_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "FGN_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
�

Kkernel
Lbias
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "FGN_out", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "FGN_out", "trainable": true, "dtype": "float32", "units": 320, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
<
^0
_1
`2
a3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
�
�layers
 �layer_regularization_losses
�	variables
�regularization_losses
�metrics
�non_trainable_variables
�trainable_variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
�
�layers
 �layer_regularization_losses
�	variables
�regularization_losses
�metrics
�non_trainable_variables
�trainable_variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
�
�layers
 �layer_regularization_losses
�	variables
�regularization_losses
�metrics
�non_trainable_variables
�trainable_variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
�
�layers
 �layer_regularization_losses
�	variables
�regularization_losses
�metrics
�non_trainable_variables
�trainable_variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
�
�layers
 �layer_regularization_losses
�	variables
�regularization_losses
�metrics
�non_trainable_variables
�trainable_variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
�
�layers
 �layer_regularization_losses
�	variables
�regularization_losses
�metrics
�non_trainable_variables
�trainable_variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
'
H0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
H0"
trackable_list_wrapper
�
�layers
 �layer_regularization_losses
�	variables
�regularization_losses
�metrics
�non_trainable_variables
�trainable_variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
I0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
I0"
trackable_list_wrapper
�
�layers
 �layer_regularization_losses
�	variables
�regularization_losses
�metrics
�non_trainable_variables
�trainable_variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
J0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
J0"
trackable_list_wrapper
�
�layers
 �layer_regularization_losses
�	variables
�regularization_losses
�metrics
�non_trainable_variables
�trainable_variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
�
�layers
 �layer_regularization_losses
�	variables
�regularization_losses
�metrics
�non_trainable_variables
�trainable_variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�2�
@__inference_model_layer_call_and_return_conditional_losses_36717
@__inference_model_layer_call_and_return_conditional_losses_37328�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
 __inference__wrapped_model_40462�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *���
���
����������
@�='�$
�������������������
�SparseTensorSpec
����������
����������	
�2�
%__inference_model_layer_call_fn_37532
%__inference_model_layer_call_fn_37446�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_ecc_conv_layer_call_and_return_conditional_losses_37063�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_ecc_conv_layer_call_fn_37169�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_gcn_conv_layer_call_and_return_conditional_losses_36874�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_gcn_conv_layer_call_fn_37084�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_gcn_conv_1_layer_call_and_return_conditional_losses_37340�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_gcn_conv_1_layer_call_fn_36840�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_gcn_conv_2_layer_call_and_return_conditional_losses_36589�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_gcn_conv_2_layer_call_fn_36895�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_gcn_conv_3_layer_call_and_return_conditional_losses_36862�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_gcn_conv_3_layer_call_fn_37190�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_global_max_pool_layer_call_and_return_conditional_losses_36536�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_global_max_pool_layer_call_fn_37096�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_signature_wrapper_40522args_0args_0_1args_0_2args_0_3args_0_4args_0_5"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
 __inference__wrapped_model_40462�HIJKL")0MNOPQRSTUVWXAB���
���
���
"�
args_0/0���������
@�='�$
�������������������
�SparseTensorSpec
"�
args_0/2���������
�
args_0/3���������	
� "3�0
.
output_1"�
output_1����������
C__inference_ecc_conv_layer_call_and_return_conditional_losses_37063�HIJKL���
���
���
"�
inputs/0���������
@�='�$
�������������������
�SparseTensorSpec
"�
inputs/2���������
� "%�"
�
0���������@
� �
(__inference_ecc_conv_layer_call_fn_37169�HIJKL���
���
���
"�
inputs/0���������
@�='�$
�������������������
�SparseTensorSpec
"�
inputs/2���������
� "����������@�
E__inference_gcn_conv_1_layer_call_and_return_conditional_losses_37340�"x�u
n�k
i�f
"�
inputs/0���������@
@�='�$
�������������������
�SparseTensorSpec
� "&�#
�
0����������
� �
*__inference_gcn_conv_1_layer_call_fn_36840�"x�u
n�k
i�f
"�
inputs/0���������@
@�='�$
�������������������
�SparseTensorSpec
� "������������
E__inference_gcn_conv_2_layer_call_and_return_conditional_losses_36589�)y�v
o�l
j�g
#� 
inputs/0����������
@�='�$
�������������������
�SparseTensorSpec
� "&�#
�
0����������
� �
*__inference_gcn_conv_2_layer_call_fn_36895�)y�v
o�l
j�g
#� 
inputs/0����������
@�='�$
�������������������
�SparseTensorSpec
� "������������
E__inference_gcn_conv_3_layer_call_and_return_conditional_losses_36862�0y�v
o�l
j�g
#� 
inputs/0����������
@�='�$
�������������������
�SparseTensorSpec
� "&�#
�
0����������
� �
*__inference_gcn_conv_3_layer_call_fn_37190�0y�v
o�l
j�g
#� 
inputs/0����������
@�='�$
�������������������
�SparseTensorSpec
� "������������
C__inference_gcn_conv_layer_call_and_return_conditional_losses_36874�x�u
n�k
i�f
"�
inputs/0���������@
@�='�$
�������������������
�SparseTensorSpec
� "%�"
�
0���������@
� �
(__inference_gcn_conv_layer_call_fn_37084�x�u
n�k
i�f
"�
inputs/0���������@
@�='�$
�������������������
�SparseTensorSpec
� "����������@�
J__inference_global_max_pool_layer_call_and_return_conditional_losses_36536�W�T
M�J
H�E
#� 
inputs/0����������
�
inputs/1���������	
� "&�#
�
0����������
� �
/__inference_global_max_pool_layer_call_fn_37096tW�T
M�J
H�E
#� 
inputs/0����������
�
inputs/1���������	
� "������������
@__inference_model_layer_call_and_return_conditional_losses_36717�HIJKL")0MNOPQRSTUVWXAB���
���
���
"�
inputs/0���������
@�='�$
�������������������
�SparseTensorSpec
"�
inputs/2���������
�
inputs/3���������	
p 
� "%�"
�
0���������
� �
@__inference_model_layer_call_and_return_conditional_losses_37328�HIJKL")0MNOPQRSTUVWXAB���
���
���
"�
inputs/0���������
@�='�$
�������������������
�SparseTensorSpec
"�
inputs/2���������
�
inputs/3���������	
p
� "%�"
�
0���������
� �
%__inference_model_layer_call_fn_37446�HIJKL")0MNOPQRSTUVWXAB���
���
���
"�
inputs/0���������
@�='�$
�������������������
�SparseTensorSpec
"�
inputs/2���������
�
inputs/3���������	
p
� "�����������
%__inference_model_layer_call_fn_37532�HIJKL")0MNOPQRSTUVWXAB���
���
���
"�
inputs/0���������
@�='�$
�������������������
�SparseTensorSpec
"�
inputs/2���������
�
inputs/3���������	
p 
� "�����������
#__inference_signature_wrapper_40522�HIJKL")0MNOPQRSTUVWXAB���
� 
���
*
args_0 �
args_0���������
.
args_0_1"�
args_0_1���������	
*
args_0_2�
args_0_2���������
!
args_0_3�
args_0_3	
.
args_0_4"�
args_0_4���������
*
args_0_5�
args_0_5���������	"3�0
.
output_1"�
output_1���������