ск
ъЗ
B
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
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
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
8
DivNoNan
x"T
y"T
z"T"
Ttype:	
2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
Г
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
\
	LeakyRelu
features"T
activations"T"
alphafloat%═╠L>"
Ttype0:
2
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
delete_old_dirsbool(ѕ
=
Mul
x"T
y"T
z"T"
Ttype:
2	љ
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
dtypetypeѕ
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
list(type)(0ѕ
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
t

SegmentMax	
data"T
segment_ids"Tindices
output"T"
Ttype:
2	"
Tindicestype:
2	
z
SegmentMean	
data"T
segment_ids"Tindices
output"T" 
Ttype:
2	"
Tindicestype:
2	
y

SegmentSum	
data"T
segment_ids"Tindices
output"T" 
Ttype:
2	"
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
╣
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
-
Sqrt
x"T
y"T"
Ttype:

2
3
Square
x"T
y"T"
Ttype:
2
	
Й
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
executor_typestring ѕ
@
StaticRegexFullMatch	
input

output
"
patternstring
Ш
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
;
Sub
x"T
y"T
z"T"
Ttype:
2	
ї
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
┴
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
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.4.02v2.4.0-2-g5485ec964e78М«
љ
model/ecc_conv/root_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*+
shared_namemodel/ecc_conv/root_kernel
Ѕ
.model/ecc_conv/root_kernel/Read/ReadVariableOpReadVariableOpmodel/ecc_conv/root_kernel*
_output_shapes

:@*
dtype0
є
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
І
model/gcn_conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@ђ*(
shared_namemodel/gcn_conv_1/kernel
ё
+model/gcn_conv_1/kernel/Read/ReadVariableOpReadVariableOpmodel/gcn_conv_1/kernel*
_output_shapes
:	@ђ*
dtype0
ї
model/gcn_conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*(
shared_namemodel/gcn_conv_2/kernel
Ё
+model/gcn_conv_2/kernel/Read/ReadVariableOpReadVariableOpmodel/gcn_conv_2/kernel* 
_output_shapes
:
ђђ*
dtype0
ї
model/gcn_conv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*(
shared_namemodel/gcn_conv_3/kernel
Ё
+model/gcn_conv_3/kernel/Read/ReadVariableOpReadVariableOpmodel/gcn_conv_3/kernel* 
_output_shapes
:
ђђ*
dtype0
Ё
model/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*%
shared_namemodel/dense_5/kernel
~
(model/dense_5/kernel/Read/ReadVariableOpReadVariableOpmodel/dense_5/kernel*
_output_shapes
:	ђ*
dtype0
|
model/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namemodel/dense_5/bias
u
&model/dense_5/bias/Read/ReadVariableOpReadVariableOpmodel/dense_5/bias*
_output_shapes
:*
dtype0
њ
model/ecc_conv/FGN_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*,
shared_namemodel/ecc_conv/FGN_0/kernel
І
/model/ecc_conv/FGN_0/kernel/Read/ReadVariableOpReadVariableOpmodel/ecc_conv/FGN_0/kernel*
_output_shapes

:@*
dtype0
њ
model/ecc_conv/FGN_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*,
shared_namemodel/ecc_conv/FGN_1/kernel
І
/model/ecc_conv/FGN_1/kernel/Read/ReadVariableOpReadVariableOpmodel/ecc_conv/FGN_1/kernel*
_output_shapes

:@@*
dtype0
њ
model/ecc_conv/FGN_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*,
shared_namemodel/ecc_conv/FGN_2/kernel
І
/model/ecc_conv/FGN_2/kernel/Read/ReadVariableOpReadVariableOpmodel/ecc_conv/FGN_2/kernel*
_output_shapes

:@@*
dtype0
Ќ
model/ecc_conv/FGN_out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@└*.
shared_namemodel/ecc_conv/FGN_out/kernel
љ
1model/ecc_conv/FGN_out/kernel/Read/ReadVariableOpReadVariableOpmodel/ecc_conv/FGN_out/kernel*
_output_shapes
:	@└*
dtype0
Ј
model/ecc_conv/FGN_out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:└*,
shared_namemodel/ecc_conv/FGN_out/bias
ѕ
/model/ecc_conv/FGN_out/bias/Read/ReadVariableOpReadVariableOpmodel/ecc_conv/FGN_out/bias*
_output_shapes	
:└*
dtype0
ѓ
model/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*#
shared_namemodel/dense/kernel
{
&model/dense/kernel/Read/ReadVariableOpReadVariableOpmodel/dense/kernel* 
_output_shapes
:
ђђ*
dtype0
y
model/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*!
shared_namemodel/dense/bias
r
$model/dense/bias/Read/ReadVariableOpReadVariableOpmodel/dense/bias*
_output_shapes	
:ђ*
dtype0
є
model/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*%
shared_namemodel/dense_1/kernel

(model/dense_1/kernel/Read/ReadVariableOpReadVariableOpmodel/dense_1/kernel* 
_output_shapes
:
ђђ*
dtype0
}
model/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*#
shared_namemodel/dense_1/bias
v
&model/dense_1/bias/Read/ReadVariableOpReadVariableOpmodel/dense_1/bias*
_output_shapes	
:ђ*
dtype0
є
model/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*%
shared_namemodel/dense_2/kernel

(model/dense_2/kernel/Read/ReadVariableOpReadVariableOpmodel/dense_2/kernel* 
_output_shapes
:
ђђ*
dtype0
}
model/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*#
shared_namemodel/dense_2/bias
v
&model/dense_2/bias/Read/ReadVariableOpReadVariableOpmodel/dense_2/bias*
_output_shapes	
:ђ*
dtype0
є
model/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*%
shared_namemodel/dense_3/kernel

(model/dense_3/kernel/Read/ReadVariableOpReadVariableOpmodel/dense_3/kernel* 
_output_shapes
:
ђђ*
dtype0
}
model/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*#
shared_namemodel/dense_3/bias
v
&model/dense_3/bias/Read/ReadVariableOpReadVariableOpmodel/dense_3/bias*
_output_shapes	
:ђ*
dtype0
є
model/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*%
shared_namemodel/dense_4/kernel

(model/dense_4/kernel/Read/ReadVariableOpReadVariableOpmodel/dense_4/kernel* 
_output_shapes
:
ђђ*
dtype0
}
model/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*#
shared_namemodel/dense_4/bias
v
&model/dense_4/bias/Read/ReadVariableOpReadVariableOpmodel/dense_4/bias*
_output_shapes	
:ђ*
dtype0
Ќ
model/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*0
shared_name!model/batch_normalization/gamma
љ
3model/batch_normalization/gamma/Read/ReadVariableOpReadVariableOpmodel/batch_normalization/gamma*
_output_shapes	
:ђ*
dtype0
Ћ
model/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*/
shared_name model/batch_normalization/beta
ј
2model/batch_normalization/beta/Read/ReadVariableOpReadVariableOpmodel/batch_normalization/beta*
_output_shapes	
:ђ*
dtype0
Џ
!model/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!model/batch_normalization_1/gamma
ћ
5model/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp!model/batch_normalization_1/gamma*
_output_shapes	
:ђ*
dtype0
Ў
 model/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*1
shared_name" model/batch_normalization_1/beta
њ
4model/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp model/batch_normalization_1/beta*
_output_shapes	
:ђ*
dtype0
Џ
!model/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!model/batch_normalization_2/gamma
ћ
5model/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp!model/batch_normalization_2/gamma*
_output_shapes	
:ђ*
dtype0
Ў
 model/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*1
shared_name" model/batch_normalization_2/beta
њ
4model/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp model/batch_normalization_2/beta*
_output_shapes	
:ђ*
dtype0
Џ
!model/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!model/batch_normalization_3/gamma
ћ
5model/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp!model/batch_normalization_3/gamma*
_output_shapes	
:ђ*
dtype0
Ў
 model/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*1
shared_name" model/batch_normalization_3/beta
њ
4model/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp model/batch_normalization_3/beta*
_output_shapes	
:ђ*
dtype0
Џ
!model/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!model/batch_normalization_4/gamma
ћ
5model/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp!model/batch_normalization_4/gamma*
_output_shapes	
:ђ*
dtype0
Ў
 model/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*1
shared_name" model/batch_normalization_4/beta
њ
4model/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp model/batch_normalization_4/beta*
_output_shapes	
:ђ*
dtype0
Б
%model/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*6
shared_name'%model/batch_normalization/moving_mean
ю
9model/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp%model/batch_normalization/moving_mean*
_output_shapes	
:ђ*
dtype0
Ф
)model/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*:
shared_name+)model/batch_normalization/moving_variance
ц
=model/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp)model/batch_normalization/moving_variance*
_output_shapes	
:ђ*
dtype0
Д
'model/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*8
shared_name)'model/batch_normalization_1/moving_mean
а
;model/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp'model/batch_normalization_1/moving_mean*
_output_shapes	
:ђ*
dtype0
»
+model/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*<
shared_name-+model/batch_normalization_1/moving_variance
е
?model/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp+model/batch_normalization_1/moving_variance*
_output_shapes	
:ђ*
dtype0
Д
'model/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*8
shared_name)'model/batch_normalization_2/moving_mean
а
;model/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp'model/batch_normalization_2/moving_mean*
_output_shapes	
:ђ*
dtype0
»
+model/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*<
shared_name-+model/batch_normalization_2/moving_variance
е
?model/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp+model/batch_normalization_2/moving_variance*
_output_shapes	
:ђ*
dtype0
Д
'model/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*8
shared_name)'model/batch_normalization_3/moving_mean
а
;model/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp'model/batch_normalization_3/moving_mean*
_output_shapes	
:ђ*
dtype0
»
+model/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*<
shared_name-+model/batch_normalization_3/moving_variance
е
?model/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp+model/batch_normalization_3/moving_variance*
_output_shapes	
:ђ*
dtype0
Д
'model/batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*8
shared_name)'model/batch_normalization_4/moving_mean
а
;model/batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp'model/batch_normalization_4/moving_mean*
_output_shapes	
:ђ*
dtype0
»
+model/batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*<
shared_name-+model/batch_normalization_4/moving_variance
е
?model/batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp+model/batch_normalization_4/moving_variance*
_output_shapes	
:ђ*
dtype0

NoOpNoOp
»i
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Жh
valueЯhBПh Bоh
┌
ECC1
GCN1
GCN2
GCN3
GCN4
	Pool1
	Pool2
	Pool3

	decode

norm_layers
d2
regularization_losses
trainable_variables
	variables
	keras_api

signatures
Б
kwargs_keys
kernel_network
kernel_network_layers
root_kernel
regularization_losses
trainable_variables
	variables
	keras_api
o
kwargs_keys

kernel
regularization_losses
trainable_variables
	variables
	keras_api
o
kwargs_keys

 kernel
!regularization_losses
"trainable_variables
#	variables
$	keras_api
o
%kwargs_keys

&kernel
'regularization_losses
(trainable_variables
)	variables
*	keras_api
o
+kwargs_keys

,kernel
-regularization_losses
.trainable_variables
/	variables
0	keras_api
R
1regularization_losses
2trainable_variables
3	variables
4	keras_api
R
5regularization_losses
6trainable_variables
7	variables
8	keras_api
R
9regularization_losses
:trainable_variables
;	variables
<	keras_api
#
=0
>1
?2
@3
A4
#
B0
C1
D2
E3
F4
h

Gkernel
Hbias
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
 
Ш
0
M1
N2
O3
P4
Q5
6
 7
&8
,9
R10
S11
T12
U13
V14
W15
X16
Y17
Z18
[19
\20
]21
^22
_23
`24
a25
b26
c27
d28
e29
G30
H31
к
0
M1
N2
O3
P4
Q5
6
 7
&8
,9
R10
S11
T12
U13
V14
W15
X16
Y17
Z18
[19
\20
]21
^22
_23
`24
a25
b26
c27
d28
e29
f30
g31
h32
i33
j34
k35
l36
m37
n38
o39
G40
H41
Г
player_regularization_losses

qlayers
regularization_losses
rmetrics
trainable_variables
slayer_metrics
tnon_trainable_variables
	variables
 
 
 

u0
v1
w2
x3
[Y
VARIABLE_VALUEmodel/ecc_conv/root_kernel+ECC1/root_kernel/.ATTRIBUTES/VARIABLE_VALUE
 
*
0
M1
N2
O3
P4
Q5
*
0
M1
N2
O3
P4
Q5
Г
ylayer_regularization_losses

zlayers
regularization_losses
{metrics
trainable_variables
|layer_metrics
}non_trainable_variables
	variables
 
QO
VARIABLE_VALUEmodel/gcn_conv/kernel&GCN1/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
░
~layer_regularization_losses

layers
regularization_losses
ђmetrics
trainable_variables
Ђlayer_metrics
ѓnon_trainable_variables
	variables
 
SQ
VARIABLE_VALUEmodel/gcn_conv_1/kernel&GCN2/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

 0

 0
▓
 Ѓlayer_regularization_losses
ёlayers
!regularization_losses
Ёmetrics
"trainable_variables
єlayer_metrics
Єnon_trainable_variables
#	variables
 
SQ
VARIABLE_VALUEmodel/gcn_conv_2/kernel&GCN3/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

&0

&0
▓
 ѕlayer_regularization_losses
Ѕlayers
'regularization_losses
іmetrics
(trainable_variables
Іlayer_metrics
їnon_trainable_variables
)	variables
 
SQ
VARIABLE_VALUEmodel/gcn_conv_3/kernel&GCN4/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

,0

,0
▓
 Їlayer_regularization_losses
јlayers
-regularization_losses
Јmetrics
.trainable_variables
љlayer_metrics
Љnon_trainable_variables
/	variables
 
 
 
▓
 њlayer_regularization_losses
Њlayers
1regularization_losses
ћmetrics
2trainable_variables
Ћlayer_metrics
ќnon_trainable_variables
3	variables
 
 
 
▓
 Ќlayer_regularization_losses
ўlayers
5regularization_losses
Ўmetrics
6trainable_variables
џlayer_metrics
Џnon_trainable_variables
7	variables
 
 
 
▓
 юlayer_regularization_losses
Юlayers
9regularization_losses
ъmetrics
:trainable_variables
Ъlayer_metrics
аnon_trainable_variables
;	variables
l

Rkernel
Sbias
Аregularization_losses
бtrainable_variables
Б	variables
ц	keras_api
l

Tkernel
Ubias
Цregularization_losses
дtrainable_variables
Д	variables
е	keras_api
l

Vkernel
Wbias
Еregularization_losses
фtrainable_variables
Ф	variables
г	keras_api
l

Xkernel
Ybias
Гregularization_losses
«trainable_variables
»	variables
░	keras_api
l

Zkernel
[bias
▒regularization_losses
▓trainable_variables
│	variables
┤	keras_api
ю
	хaxis
	\gamma
]beta
fmoving_mean
gmoving_variance
Хregularization_losses
иtrainable_variables
И	variables
╣	keras_api
ю
	║axis
	^gamma
_beta
hmoving_mean
imoving_variance
╗regularization_losses
╝trainable_variables
й	variables
Й	keras_api
ю
	┐axis
	`gamma
abeta
jmoving_mean
kmoving_variance
└regularization_losses
┴trainable_variables
┬	variables
├	keras_api
ю
	─axis
	bgamma
cbeta
lmoving_mean
mmoving_variance
┼regularization_losses
кtrainable_variables
К	variables
╚	keras_api
ю
	╔axis
	dgamma
ebeta
nmoving_mean
omoving_variance
╩regularization_losses
╦trainable_variables
╠	variables
═	keras_api
NL
VARIABLE_VALUEmodel/dense_5/kernel$d2/kernel/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEmodel/dense_5/bias"d2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

G0
H1

G0
H1
▓
 ╬layer_regularization_losses
¤layers
Iregularization_losses
лmetrics
Jtrainable_variables
Лlayer_metrics
мnon_trainable_variables
K	variables
a_
VARIABLE_VALUEmodel/ecc_conv/FGN_0/kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEmodel/ecc_conv/FGN_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEmodel/ecc_conv/FGN_2/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEmodel/ecc_conv/FGN_out/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEmodel/ecc_conv/FGN_out/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEmodel/dense/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEmodel/dense/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEmodel/dense_1/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEmodel/dense_1/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEmodel/dense_2/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEmodel/dense_2/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEmodel/dense_3/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEmodel/dense_3/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEmodel/dense_4/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEmodel/dense_4/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEmodel/batch_normalization/gamma1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEmodel/batch_normalization/beta1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!model/batch_normalization_1/gamma1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE model/batch_normalization_1/beta1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!model/batch_normalization_2/gamma1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE model/batch_normalization_2/beta1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!model/batch_normalization_3/gamma1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE model/batch_normalization_3/beta1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!model/batch_normalization_4/gamma1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE model/batch_normalization_4/beta1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%model/batch_normalization/moving_mean'variables/30/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)model/batch_normalization/moving_variance'variables/31/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'model/batch_normalization_1/moving_mean'variables/32/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+model/batch_normalization_1/moving_variance'variables/33/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'model/batch_normalization_2/moving_mean'variables/34/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+model/batch_normalization_2/moving_variance'variables/35/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'model/batch_normalization_3/moving_mean'variables/36/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+model/batch_normalization_3/moving_variance'variables/37/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'model/batch_normalization_4/moving_mean'variables/38/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+model/batch_normalization_4/moving_variance'variables/39/.ATTRIBUTES/VARIABLE_VALUE
 
ј
0
1
2
3
4
5
6
7
=8
>9
?10
@11
A12
B13
C14
D15
E16
F17
18
 
 
F
f0
g1
h2
i3
j4
k5
l6
m7
n8
o9
b

Mkernel
Мregularization_losses
нtrainable_variables
Н	variables
о	keras_api
b

Nkernel
Оregularization_losses
пtrainable_variables
┘	variables
┌	keras_api
b

Okernel
█regularization_losses
▄trainable_variables
П	variables
я	keras_api
l

Pkernel
Qbias
▀regularization_losses
Яtrainable_variables
р	variables
Р	keras_api
 

u0
v1
w2
x3
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

R0
S1

R0
S1
х
 сlayer_regularization_losses
Сlayers
Аregularization_losses
тmetrics
бtrainable_variables
Тlayer_metrics
уnon_trainable_variables
Б	variables
 

T0
U1

T0
U1
х
 Уlayer_regularization_losses
жlayers
Цregularization_losses
Жmetrics
дtrainable_variables
вlayer_metrics
Вnon_trainable_variables
Д	variables
 

V0
W1

V0
W1
х
 ьlayer_regularization_losses
Ьlayers
Еregularization_losses
№metrics
фtrainable_variables
­layer_metrics
ыnon_trainable_variables
Ф	variables
 

X0
Y1

X0
Y1
х
 Ыlayer_regularization_losses
зlayers
Гregularization_losses
Зmetrics
«trainable_variables
шlayer_metrics
Шnon_trainable_variables
»	variables
 

Z0
[1

Z0
[1
х
 эlayer_regularization_losses
Эlayers
▒regularization_losses
щmetrics
▓trainable_variables
Щlayer_metrics
чnon_trainable_variables
│	variables
 
 

\0
]1

\0
]1
f2
g3
х
 Чlayer_regularization_losses
§layers
Хregularization_losses
■metrics
иtrainable_variables
 layer_metrics
ђnon_trainable_variables
И	variables
 
 

^0
_1

^0
_1
h2
i3
х
 Ђlayer_regularization_losses
ѓlayers
╗regularization_losses
Ѓmetrics
╝trainable_variables
ёlayer_metrics
Ёnon_trainable_variables
й	variables
 
 

`0
a1

`0
a1
j2
k3
х
 єlayer_regularization_losses
Єlayers
└regularization_losses
ѕmetrics
┴trainable_variables
Ѕlayer_metrics
іnon_trainable_variables
┬	variables
 
 

b0
c1

b0
c1
l2
m3
х
 Іlayer_regularization_losses
їlayers
┼regularization_losses
Їmetrics
кtrainable_variables
јlayer_metrics
Јnon_trainable_variables
К	variables
 
 

d0
e1

d0
e1
n2
o3
х
 љlayer_regularization_losses
Љlayers
╩regularization_losses
њmetrics
╦trainable_variables
Њlayer_metrics
ћnon_trainable_variables
╠	variables
 
 
 
 
 
 

M0

M0
х
 Ћlayer_regularization_losses
ќlayers
Мregularization_losses
Ќmetrics
нtrainable_variables
ўlayer_metrics
Ўnon_trainable_variables
Н	variables
 

N0

N0
х
 џlayer_regularization_losses
Џlayers
Оregularization_losses
юmetrics
пtrainable_variables
Юlayer_metrics
ъnon_trainable_variables
┘	variables
 

O0

O0
х
 Ъlayer_regularization_losses
аlayers
█regularization_losses
Аmetrics
▄trainable_variables
бlayer_metrics
Бnon_trainable_variables
П	variables
 

P0
Q1

P0
Q1
х
 цlayer_regularization_losses
Цlayers
▀regularization_losses
дmetrics
Яtrainable_variables
Дlayer_metrics
еnon_trainable_variables
р	variables
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
f0
g1
 
 
 
 

h0
i1
 
 
 
 

j0
k1
 
 
 
 

l0
m1
 
 
 
 

n0
o1
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
:         *
dtype0*
shape:         
{
serving_default_args_0_1Placeholder*'
_output_shapes
:         *
dtype0	*
shape:         
s
serving_default_args_0_2Placeholder*#
_output_shapes
:         *
dtype0*
shape:         
a
serving_default_args_0_3Placeholder*
_output_shapes
:*
dtype0	*
shape:
s
serving_default_args_0_4Placeholder*#
_output_shapes
:         *
dtype0	*
shape:         
§
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_0_1serving_default_args_0_2serving_default_args_0_3serving_default_args_0_4model/ecc_conv/FGN_0/kernelmodel/ecc_conv/FGN_1/kernelmodel/ecc_conv/FGN_2/kernelmodel/ecc_conv/FGN_out/kernelmodel/ecc_conv/FGN_out/biasmodel/ecc_conv/root_kernelmodel/gcn_conv/kernelmodel/gcn_conv_1/kernelmodel/gcn_conv_2/kernelmodel/gcn_conv_3/kernelmodel/dense/kernelmodel/dense/bias%model/batch_normalization/moving_mean)model/batch_normalization/moving_variancemodel/batch_normalization/betamodel/batch_normalization/gammamodel/dense_1/kernelmodel/dense_1/bias'model/batch_normalization_1/moving_mean+model/batch_normalization_1/moving_variance model/batch_normalization_1/beta!model/batch_normalization_1/gammamodel/dense_2/kernelmodel/dense_2/bias'model/batch_normalization_2/moving_mean+model/batch_normalization_2/moving_variance model/batch_normalization_2/beta!model/batch_normalization_2/gammamodel/dense_3/kernelmodel/dense_3/bias'model/batch_normalization_3/moving_mean+model/batch_normalization_3/moving_variance model/batch_normalization_3/beta!model/batch_normalization_3/gammamodel/dense_4/kernelmodel/dense_4/bias'model/batch_normalization_4/moving_mean+model/batch_normalization_4/moving_variance model/batch_normalization_4/beta!model/batch_normalization_4/gammamodel/dense_5/kernelmodel/dense_5/bias*:
Tin3
12/			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *L
_read_only_resource_inputs.
,*	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8ѓ *,
f'R%
#__inference_signature_wrapper_62326
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ў
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename.model/ecc_conv/root_kernel/Read/ReadVariableOp)model/gcn_conv/kernel/Read/ReadVariableOp+model/gcn_conv_1/kernel/Read/ReadVariableOp+model/gcn_conv_2/kernel/Read/ReadVariableOp+model/gcn_conv_3/kernel/Read/ReadVariableOp(model/dense_5/kernel/Read/ReadVariableOp&model/dense_5/bias/Read/ReadVariableOp/model/ecc_conv/FGN_0/kernel/Read/ReadVariableOp/model/ecc_conv/FGN_1/kernel/Read/ReadVariableOp/model/ecc_conv/FGN_2/kernel/Read/ReadVariableOp1model/ecc_conv/FGN_out/kernel/Read/ReadVariableOp/model/ecc_conv/FGN_out/bias/Read/ReadVariableOp&model/dense/kernel/Read/ReadVariableOp$model/dense/bias/Read/ReadVariableOp(model/dense_1/kernel/Read/ReadVariableOp&model/dense_1/bias/Read/ReadVariableOp(model/dense_2/kernel/Read/ReadVariableOp&model/dense_2/bias/Read/ReadVariableOp(model/dense_3/kernel/Read/ReadVariableOp&model/dense_3/bias/Read/ReadVariableOp(model/dense_4/kernel/Read/ReadVariableOp&model/dense_4/bias/Read/ReadVariableOp3model/batch_normalization/gamma/Read/ReadVariableOp2model/batch_normalization/beta/Read/ReadVariableOp5model/batch_normalization_1/gamma/Read/ReadVariableOp4model/batch_normalization_1/beta/Read/ReadVariableOp5model/batch_normalization_2/gamma/Read/ReadVariableOp4model/batch_normalization_2/beta/Read/ReadVariableOp5model/batch_normalization_3/gamma/Read/ReadVariableOp4model/batch_normalization_3/beta/Read/ReadVariableOp5model/batch_normalization_4/gamma/Read/ReadVariableOp4model/batch_normalization_4/beta/Read/ReadVariableOp9model/batch_normalization/moving_mean/Read/ReadVariableOp=model/batch_normalization/moving_variance/Read/ReadVariableOp;model/batch_normalization_1/moving_mean/Read/ReadVariableOp?model/batch_normalization_1/moving_variance/Read/ReadVariableOp;model/batch_normalization_2/moving_mean/Read/ReadVariableOp?model/batch_normalization_2/moving_variance/Read/ReadVariableOp;model/batch_normalization_3/moving_mean/Read/ReadVariableOp?model/batch_normalization_3/moving_variance/Read/ReadVariableOp;model/batch_normalization_4/moving_mean/Read/ReadVariableOp?model/batch_normalization_4/moving_variance/Read/ReadVariableOpConst*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *'
f"R 
__inference__traced_save_63952
╠
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemodel/ecc_conv/root_kernelmodel/gcn_conv/kernelmodel/gcn_conv_1/kernelmodel/gcn_conv_2/kernelmodel/gcn_conv_3/kernelmodel/dense_5/kernelmodel/dense_5/biasmodel/ecc_conv/FGN_0/kernelmodel/ecc_conv/FGN_1/kernelmodel/ecc_conv/FGN_2/kernelmodel/ecc_conv/FGN_out/kernelmodel/ecc_conv/FGN_out/biasmodel/dense/kernelmodel/dense/biasmodel/dense_1/kernelmodel/dense_1/biasmodel/dense_2/kernelmodel/dense_2/biasmodel/dense_3/kernelmodel/dense_3/biasmodel/dense_4/kernelmodel/dense_4/biasmodel/batch_normalization/gammamodel/batch_normalization/beta!model/batch_normalization_1/gamma model/batch_normalization_1/beta!model/batch_normalization_2/gamma model/batch_normalization_2/beta!model/batch_normalization_3/gamma model/batch_normalization_3/beta!model/batch_normalization_4/gamma model/batch_normalization_4/beta%model/batch_normalization/moving_mean)model/batch_normalization/moving_variance'model/batch_normalization_1/moving_mean+model/batch_normalization_1/moving_variance'model/batch_normalization_2/moving_mean+model/batch_normalization_2/moving_variance'model/batch_normalization_3/moving_mean+model/batch_normalization_3/moving_variance'model/batch_normalization_4/moving_mean+model/batch_normalization_4/moving_variance*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ **
f%R#
!__inference__traced_restore_64088к▒
ќ
┘
N__inference_batch_normalization_layer_call_and_return_conditional_losses_63445

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identityѕбCast/ReadVariableOpбCast_1/ReadVariableOpбCast_2/ReadVariableOpбCast_3/ReadVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_1/ReadVariableOpі
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_2/ReadVariableOpі
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yє
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/add_1к
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ў	
█
B__inference_dense_4_layer_call_and_return_conditional_losses_63380

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddќ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
гО
І
@__inference_model_layer_call_and_return_conditional_losses_62886
inputs_0

inputs	
inputs_1
inputs_2	

inputs_2_0	1
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
%dense_biasadd_readvariableop_resource4
0batch_normalization_cast_readvariableop_resource6
2batch_normalization_cast_1_readvariableop_resource6
2batch_normalization_cast_2_readvariableop_resource6
2batch_normalization_cast_3_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource6
2batch_normalization_1_cast_readvariableop_resource8
4batch_normalization_1_cast_1_readvariableop_resource8
4batch_normalization_1_cast_2_readvariableop_resource8
4batch_normalization_1_cast_3_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource6
2batch_normalization_2_cast_readvariableop_resource8
4batch_normalization_2_cast_1_readvariableop_resource8
4batch_normalization_2_cast_2_readvariableop_resource8
4batch_normalization_2_cast_3_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource6
2batch_normalization_3_cast_readvariableop_resource8
4batch_normalization_3_cast_1_readvariableop_resource8
4batch_normalization_3_cast_2_readvariableop_resource8
4batch_normalization_3_cast_3_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource6
2batch_normalization_4_cast_readvariableop_resource8
4batch_normalization_4_cast_1_readvariableop_resource8
4batch_normalization_4_cast_2_readvariableop_resource8
4batch_normalization_4_cast_3_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identityѕб'batch_normalization/Cast/ReadVariableOpб)batch_normalization/Cast_1/ReadVariableOpб)batch_normalization/Cast_2/ReadVariableOpб)batch_normalization/Cast_3/ReadVariableOpб)batch_normalization_1/Cast/ReadVariableOpб+batch_normalization_1/Cast_1/ReadVariableOpб+batch_normalization_1/Cast_2/ReadVariableOpб+batch_normalization_1/Cast_3/ReadVariableOpб)batch_normalization_2/Cast/ReadVariableOpб+batch_normalization_2/Cast_1/ReadVariableOpб+batch_normalization_2/Cast_2/ReadVariableOpб+batch_normalization_2/Cast_3/ReadVariableOpб)batch_normalization_3/Cast/ReadVariableOpб+batch_normalization_3/Cast_1/ReadVariableOpб+batch_normalization_3/Cast_2/ReadVariableOpб+batch_normalization_3/Cast_3/ReadVariableOpб)batch_normalization_4/Cast/ReadVariableOpб+batch_normalization_4/Cast_1/ReadVariableOpб+batch_normalization_4/Cast_2/ReadVariableOpб+batch_normalization_4/Cast_3/ReadVariableOpбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpбdense_2/BiasAdd/ReadVariableOpбdense_2/MatMul/ReadVariableOpбdense_3/BiasAdd/ReadVariableOpбdense_3/MatMul/ReadVariableOpбdense_4/BiasAdd/ReadVariableOpбdense_4/MatMul/ReadVariableOpбdense_5/BiasAdd/ReadVariableOpбdense_5/MatMul/ReadVariableOpб$ecc_conv/FGN_0/MatMul/ReadVariableOpб$ecc_conv/FGN_1/MatMul/ReadVariableOpб$ecc_conv/FGN_2/MatMul/ReadVariableOpб'ecc_conv/FGN_out/BiasAdd/ReadVariableOpб&ecc_conv/FGN_out/MatMul/ReadVariableOpб ecc_conv/MatMul_1/ReadVariableOpбgcn_conv/MatMul/ReadVariableOpб gcn_conv_1/MatMul/ReadVariableOpб gcn_conv_2/MatMul/ReadVariableOpб gcn_conv_3/MatMul/ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2Ѕ
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stackЃ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1Ѓ
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2Њ
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis▒
GatherV2GatherV2inputs_0strided_slice_1:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:         2

GatherV2d
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_1/axisх

GatherV2_1GatherV2inputs_0strided_slice:output:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:         2

GatherV2_1k
SubSubGatherV2:output:0GatherV2_1:output:0*
T0*'
_output_shapes
:         2
Sub
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stackЃ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1Ѓ
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2ђ
strided_slice_2StridedSliceSub:z:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
strided_slice_2f
SquareSquarestrided_slice_2:output:0*
T0*'
_output_shapes
:         2
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesk
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
SumP
SqrtSqrtSum:output:0*
T0*#
_output_shapes
:         2
Sqrt
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stackЃ
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1Ѓ
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2ђ
strided_slice_3StridedSliceSub:z:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
strided_slice_3k
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         2
ExpandDims/dim{

ExpandDims
ExpandDimsSqrt:y:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:         2

ExpandDimsЁ

div_no_nanDivNoNanstrided_slice_3:output:0ExpandDims:output:0*
T0*'
_output_shapes
:         2

div_no_nan
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stackЃ
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_4/stack_1Ѓ
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2ђ
strided_slice_4StridedSliceSub:z:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
strided_slice_4o
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
         2
ExpandDims_1/dimЂ
ExpandDims_1
ExpandDimsSqrt:y:0ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:         2
ExpandDims_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis«
concatConcatV2strided_slice_4:output:0ExpandDims_1:output:0div_no_nan:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
concatX
ecc_conv/ShapeShapeinputs_0*
T0*
_output_shapes
:2
ecc_conv/ShapeЈ
ecc_conv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        2
ecc_conv/strided_slice/stackЊ
ecc_conv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         2 
ecc_conv/strided_slice/stack_1і
ecc_conv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
ecc_conv/strided_slice/stack_2ў
ecc_conv/strided_sliceStridedSliceecc_conv/Shape:output:0%ecc_conv/strided_slice/stack:output:0'ecc_conv/strided_slice/stack_1:output:0'ecc_conv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ecc_conv/strided_slice║
$ecc_conv/FGN_0/MatMul/ReadVariableOpReadVariableOp-ecc_conv_fgn_0_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02&
$ecc_conv/FGN_0/MatMul/ReadVariableOpЕ
ecc_conv/FGN_0/MatMulMatMulconcat:output:0,ecc_conv/FGN_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
ecc_conv/FGN_0/MatMulЁ
ecc_conv/FGN_0/ReluReluecc_conv/FGN_0/MatMul:product:0*
T0*'
_output_shapes
:         @2
ecc_conv/FGN_0/Relu║
$ecc_conv/FGN_1/MatMul/ReadVariableOpReadVariableOp-ecc_conv_fgn_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02&
$ecc_conv/FGN_1/MatMul/ReadVariableOp╗
ecc_conv/FGN_1/MatMulMatMul!ecc_conv/FGN_0/Relu:activations:0,ecc_conv/FGN_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
ecc_conv/FGN_1/MatMulЁ
ecc_conv/FGN_1/ReluReluecc_conv/FGN_1/MatMul:product:0*
T0*'
_output_shapes
:         @2
ecc_conv/FGN_1/Relu║
$ecc_conv/FGN_2/MatMul/ReadVariableOpReadVariableOp-ecc_conv_fgn_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02&
$ecc_conv/FGN_2/MatMul/ReadVariableOp╗
ecc_conv/FGN_2/MatMulMatMul!ecc_conv/FGN_1/Relu:activations:0,ecc_conv/FGN_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
ecc_conv/FGN_2/MatMulЁ
ecc_conv/FGN_2/ReluReluecc_conv/FGN_2/MatMul:product:0*
T0*'
_output_shapes
:         @2
ecc_conv/FGN_2/Relu┴
&ecc_conv/FGN_out/MatMul/ReadVariableOpReadVariableOp/ecc_conv_fgn_out_matmul_readvariableop_resource*
_output_shapes
:	@└*
dtype02(
&ecc_conv/FGN_out/MatMul/ReadVariableOp┬
ecc_conv/FGN_out/MatMulMatMul!ecc_conv/FGN_2/Relu:activations:0.ecc_conv/FGN_out/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └2
ecc_conv/FGN_out/MatMul└
'ecc_conv/FGN_out/BiasAdd/ReadVariableOpReadVariableOp0ecc_conv_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:└*
dtype02)
'ecc_conv/FGN_out/BiasAdd/ReadVariableOpк
ecc_conv/FGN_out/BiasAddBiasAdd!ecc_conv/FGN_out/MatMul:product:0/ecc_conv/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └2
ecc_conv/FGN_out/BiasAddЁ
ecc_conv/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       @   2
ecc_conv/Reshape/shapeЕ
ecc_conv/ReshapeReshape!ecc_conv/FGN_out/BiasAdd:output:0ecc_conv/Reshape/shape:output:0*
T0*+
_output_shapes
:         @2
ecc_conv/ReshapeЉ
ecc_conv/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ■   2 
ecc_conv/strided_slice_1/stackЋ
 ecc_conv/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 ecc_conv/strided_slice_1/stack_1Ћ
 ecc_conv/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 ecc_conv/strided_slice_1/stack_2└
ecc_conv/strided_slice_1StridedSliceinputs'ecc_conv/strided_slice_1/stack:output:0)ecc_conv/strided_slice_1/stack_1:output:0)ecc_conv/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
ecc_conv/strided_slice_1Љ
ecc_conv/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
ecc_conv/strided_slice_2/stackЋ
 ecc_conv/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 ecc_conv/strided_slice_2/stack_1Ћ
 ecc_conv/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 ecc_conv/strided_slice_2/stack_2└
ecc_conv/strided_slice_2StridedSliceinputs'ecc_conv/strided_slice_2/stack:output:0)ecc_conv/strided_slice_2/stack_1:output:0)ecc_conv/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
ecc_conv/strided_slice_2r
ecc_conv/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
ecc_conv/GatherV2/axisН
ecc_conv/GatherV2GatherV2inputs_0!ecc_conv/strided_slice_2:output:0ecc_conv/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:         2
ecc_conv/GatherV2Ћ
ecc_conv/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2 
ecc_conv/strided_slice_3/stackЎ
 ecc_conv/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2"
 ecc_conv/strided_slice_3/stack_1Ў
 ecc_conv/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2"
 ecc_conv/strided_slice_3/stack_2┘
ecc_conv/strided_slice_3StridedSliceecc_conv/GatherV2:output:0'ecc_conv/strided_slice_3/stack:output:0)ecc_conv/strided_slice_3/stack_1:output:0)ecc_conv/strided_slice_3/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2
ecc_conv/strided_slice_3Д
ecc_conv/MatMulBatchMatMulV2!ecc_conv/strided_slice_3:output:0ecc_conv/Reshape:output:0*
T0*+
_output_shapes
:         @2
ecc_conv/MatMulЋ
ecc_conv/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2 
ecc_conv/strided_slice_4/stackЎ
 ecc_conv/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2"
 ecc_conv/strided_slice_4/stack_1Ў
 ecc_conv/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2"
 ecc_conv/strided_slice_4/stack_2о
ecc_conv/strided_slice_4StridedSliceecc_conv/MatMul:output:0'ecc_conv/strided_slice_4/stack:output:0)ecc_conv/strided_slice_4/stack_1:output:0)ecc_conv/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*

begin_mask*
end_mask*
shrink_axis_mask2
ecc_conv/strided_slice_4щ
ecc_conv/UnsortedSegmentSumUnsortedSegmentSum!ecc_conv/strided_slice_4:output:0!ecc_conv/strided_slice_1:output:0ecc_conv/strided_slice:output:0*
T0*
Tindices0	*'
_output_shapes
:         @2
ecc_conv/UnsortedSegmentSum«
 ecc_conv/MatMul_1/ReadVariableOpReadVariableOp)ecc_conv_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02"
 ecc_conv/MatMul_1/ReadVariableOpќ
ecc_conv/MatMul_1MatMulinputs_0(ecc_conv/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
ecc_conv/MatMul_1џ
ecc_conv/addAddV2$ecc_conv/UnsortedSegmentSum:output:0ecc_conv/MatMul_1:product:0*
T0*'
_output_shapes
:         @2
ecc_conv/addj
ecc_conv/ReluReluecc_conv/add:z:0*
T0*'
_output_shapes
:         @2
ecc_conv/Reluе
gcn_conv/MatMul/ReadVariableOpReadVariableOp'gcn_conv_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
gcn_conv/MatMul/ReadVariableOpБ
gcn_conv/MatMulMatMulecc_conv/Relu:activations:0&gcn_conv/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
gcn_conv/MatMulЭ
8gcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2gcn_conv/MatMul:product:0*
T0*'
_output_shapes
:         @2:
8gcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMulю
gcn_conv/ReluReluBgcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*'
_output_shapes
:         @2
gcn_conv/Relu»
 gcn_conv_1/MatMul/ReadVariableOpReadVariableOp)gcn_conv_1_matmul_readvariableop_resource*
_output_shapes
:	@ђ*
dtype02"
 gcn_conv_1/MatMul/ReadVariableOpф
gcn_conv_1/MatMulMatMulgcn_conv/Relu:activations:0(gcn_conv_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
gcn_conv_1/MatMul 
:gcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2gcn_conv_1/MatMul:product:0*
T0*(
_output_shapes
:         ђ2<
:gcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMulБ
gcn_conv_1/ReluReluDgcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:         ђ2
gcn_conv_1/Relu░
 gcn_conv_2/MatMul/ReadVariableOpReadVariableOp)gcn_conv_2_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02"
 gcn_conv_2/MatMul/ReadVariableOpг
gcn_conv_2/MatMulMatMulgcn_conv_1/Relu:activations:0(gcn_conv_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
gcn_conv_2/MatMul 
:gcn_conv_2/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2gcn_conv_2/MatMul:product:0*
T0*(
_output_shapes
:         ђ2<
:gcn_conv_2/SparseTensorDenseMatMul/SparseTensorDenseMatMulБ
gcn_conv_2/ReluReluDgcn_conv_2/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:         ђ2
gcn_conv_2/Relu░
 gcn_conv_3/MatMul/ReadVariableOpReadVariableOp)gcn_conv_3_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02"
 gcn_conv_3/MatMul/ReadVariableOpг
gcn_conv_3/MatMulMatMulgcn_conv_2/Relu:activations:0(gcn_conv_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
gcn_conv_3/MatMul 
:gcn_conv_3/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2gcn_conv_3/MatMul:product:0*
T0*(
_output_shapes
:         ђ2<
:gcn_conv_3/SparseTensorDenseMatMul/SparseTensorDenseMatMulБ
gcn_conv_3/ReluReluDgcn_conv_3/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:         ђ2
gcn_conv_3/Relu┤
global_max_pool/SegmentMax
SegmentMaxgcn_conv_3/Relu:activations:0
inputs_2_0*
T0*
Tindices0	*(
_output_shapes
:         ђ2
global_max_pool/SegmentMaxи
global_avg_pool/SegmentMeanSegmentMeangcn_conv_3/Relu:activations:0
inputs_2_0*
T0*
Tindices0	*(
_output_shapes
:         ђ2
global_avg_pool/SegmentMean┤
global_sum_pool/SegmentSum
SegmentSumgcn_conv_3/Relu:activations:0
inputs_2_0*
T0*
Tindices0	*(
_output_shapes
:         ђ2
global_sum_pool/SegmentSum`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axisС
concat_1ConcatV2#global_max_pool/SegmentMax:output:0$global_avg_pool/SegmentMean:output:0#global_sum_pool/SegmentSum:output:0concat_1/axis:output:0*
N*
T0*(
_output_shapes
:         ђ2

concat_1А
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
dense/MatMul/ReadVariableOpЉ
dense/MatMulMatMulconcat_1:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense/MatMulЪ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
dense/BiasAdd/ReadVariableOpџ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense/BiasAddЇ
leaky_re_lu/LeakyRelu	LeakyReludense/BiasAdd:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
leaky_re_lu/LeakyRelu└
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02)
'batch_normalization/Cast/ReadVariableOpк
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)batch_normalization/Cast_1/ReadVariableOpк
)batch_normalization/Cast_2/ReadVariableOpReadVariableOp2batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)batch_normalization/Cast_2/ReadVariableOpк
)batch_normalization/Cast_3/ReadVariableOpReadVariableOp2batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)batch_normalization/Cast_3/ReadVariableOpЈ
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2%
#batch_normalization/batchnorm/add/yо
!batch_normalization/batchnorm/addAddV21batch_normalization/Cast_1/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2#
!batch_normalization/batchnorm/addа
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization/batchnorm/Rsqrt¤
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2#
!batch_normalization/batchnorm/mulл
#batch_normalization/batchnorm/mul_1Mul#leaky_re_lu/LeakyRelu:activations:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2%
#batch_normalization/batchnorm/mul_1¤
#batch_normalization/batchnorm/mul_2Mul/batch_normalization/Cast/ReadVariableOp:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization/batchnorm/mul_2¤
!batch_normalization/batchnorm/subSub1batch_normalization/Cast_2/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2#
!batch_normalization/batchnorm/subо
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2%
#batch_normalization/batchnorm/add_1Д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
dense_1/MatMul/ReadVariableOpГ
dense_1/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_1/MatMulЦ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_1/BiasAddЊ
leaky_re_lu/LeakyRelu_1	LeakyReludense_1/BiasAdd:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
leaky_re_lu/LeakyRelu_1к
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)batch_normalization_1/Cast/ReadVariableOp╠
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+batch_normalization_1/Cast_1/ReadVariableOp╠
+batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+batch_normalization_1/Cast_2/ReadVariableOp╠
+batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+batch_normalization_1/Cast_3/ReadVariableOpЊ
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_1/batchnorm/add/yя
#batch_normalization_1/batchnorm/addAddV23batch_normalization_1/Cast_1/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_1/batchnorm/addд
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_1/batchnorm/RsqrtО
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_1/batchnorm/mulп
%batch_normalization_1/batchnorm/mul_1Mul%leaky_re_lu/LeakyRelu_1:activations:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2'
%batch_normalization_1/batchnorm/mul_1О
%batch_normalization_1/batchnorm/mul_2Mul1batch_normalization_1/Cast/ReadVariableOp:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_1/batchnorm/mul_2О
#batch_normalization_1/batchnorm/subSub3batch_normalization_1/Cast_2/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_1/batchnorm/subя
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2'
%batch_normalization_1/batchnorm/add_1Д
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
dense_2/MatMul/ReadVariableOp»
dense_2/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_2/MatMulЦ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
dense_2/BiasAdd/ReadVariableOpб
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_2/BiasAddЊ
leaky_re_lu/LeakyRelu_2	LeakyReludense_2/BiasAdd:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
leaky_re_lu/LeakyRelu_2к
)batch_normalization_2/Cast/ReadVariableOpReadVariableOp2batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)batch_normalization_2/Cast/ReadVariableOp╠
+batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+batch_normalization_2/Cast_1/ReadVariableOp╠
+batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+batch_normalization_2/Cast_2/ReadVariableOp╠
+batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+batch_normalization_2/Cast_3/ReadVariableOpЊ
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_2/batchnorm/add/yя
#batch_normalization_2/batchnorm/addAddV23batch_normalization_2/Cast_1/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_2/batchnorm/addд
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_2/batchnorm/RsqrtО
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:03batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_2/batchnorm/mulп
%batch_normalization_2/batchnorm/mul_1Mul%leaky_re_lu/LeakyRelu_2:activations:0'batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2'
%batch_normalization_2/batchnorm/mul_1О
%batch_normalization_2/batchnorm/mul_2Mul1batch_normalization_2/Cast/ReadVariableOp:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_2/batchnorm/mul_2О
#batch_normalization_2/batchnorm/subSub3batch_normalization_2/Cast_2/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_2/batchnorm/subя
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2'
%batch_normalization_2/batchnorm/add_1Д
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
dense_3/MatMul/ReadVariableOp»
dense_3/MatMulMatMul)batch_normalization_2/batchnorm/add_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_3/MatMulЦ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
dense_3/BiasAdd/ReadVariableOpб
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_3/BiasAddЊ
leaky_re_lu/LeakyRelu_3	LeakyReludense_3/BiasAdd:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
leaky_re_lu/LeakyRelu_3к
)batch_normalization_3/Cast/ReadVariableOpReadVariableOp2batch_normalization_3_cast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)batch_normalization_3/Cast/ReadVariableOp╠
+batch_normalization_3/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_3_cast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+batch_normalization_3/Cast_1/ReadVariableOp╠
+batch_normalization_3/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_3_cast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+batch_normalization_3/Cast_2/ReadVariableOp╠
+batch_normalization_3/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_3_cast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+batch_normalization_3/Cast_3/ReadVariableOpЊ
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_3/batchnorm/add/yя
#batch_normalization_3/batchnorm/addAddV23batch_normalization_3/Cast_1/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_3/batchnorm/addд
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_3/batchnorm/RsqrtО
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:03batch_normalization_3/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_3/batchnorm/mulп
%batch_normalization_3/batchnorm/mul_1Mul%leaky_re_lu/LeakyRelu_3:activations:0'batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2'
%batch_normalization_3/batchnorm/mul_1О
%batch_normalization_3/batchnorm/mul_2Mul1batch_normalization_3/Cast/ReadVariableOp:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_3/batchnorm/mul_2О
#batch_normalization_3/batchnorm/subSub3batch_normalization_3/Cast_2/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_3/batchnorm/subя
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2'
%batch_normalization_3/batchnorm/add_1Д
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
dense_4/MatMul/ReadVariableOp»
dense_4/MatMulMatMul)batch_normalization_3/batchnorm/add_1:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_4/MatMulЦ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
dense_4/BiasAdd/ReadVariableOpб
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_4/BiasAddЊ
leaky_re_lu/LeakyRelu_4	LeakyReludense_4/BiasAdd:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
leaky_re_lu/LeakyRelu_4к
)batch_normalization_4/Cast/ReadVariableOpReadVariableOp2batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)batch_normalization_4/Cast/ReadVariableOp╠
+batch_normalization_4/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+batch_normalization_4/Cast_1/ReadVariableOp╠
+batch_normalization_4/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_4_cast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+batch_normalization_4/Cast_2/ReadVariableOp╠
+batch_normalization_4/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_4_cast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+batch_normalization_4/Cast_3/ReadVariableOpЊ
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_4/batchnorm/add/yя
#batch_normalization_4/batchnorm/addAddV23batch_normalization_4/Cast_1/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_4/batchnorm/addд
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_4/batchnorm/RsqrtО
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:03batch_normalization_4/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_4/batchnorm/mulп
%batch_normalization_4/batchnorm/mul_1Mul%leaky_re_lu/LeakyRelu_4:activations:0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2'
%batch_normalization_4/batchnorm/mul_1О
%batch_normalization_4/batchnorm/mul_2Mul1batch_normalization_4/Cast/ReadVariableOp:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_4/batchnorm/mul_2О
#batch_normalization_4/batchnorm/subSub3batch_normalization_4/Cast_2/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_4/batchnorm/subя
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2'
%batch_normalization_4/batchnorm/add_1д
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
dense_5/MatMul/ReadVariableOp«
dense_5/MatMulMatMul)batch_normalization_4/batchnorm/add_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/MatMulц
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOpА
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/BiasAddж
IdentityIdentitydense_5/BiasAdd:output:0(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp*^batch_normalization/Cast_2/ReadVariableOp*^batch_normalization/Cast_3/ReadVariableOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp,^batch_normalization_1/Cast_2/ReadVariableOp,^batch_normalization_1/Cast_3/ReadVariableOp*^batch_normalization_2/Cast/ReadVariableOp,^batch_normalization_2/Cast_1/ReadVariableOp,^batch_normalization_2/Cast_2/ReadVariableOp,^batch_normalization_2/Cast_3/ReadVariableOp*^batch_normalization_3/Cast/ReadVariableOp,^batch_normalization_3/Cast_1/ReadVariableOp,^batch_normalization_3/Cast_2/ReadVariableOp,^batch_normalization_3/Cast_3/ReadVariableOp*^batch_normalization_4/Cast/ReadVariableOp,^batch_normalization_4/Cast_1/ReadVariableOp,^batch_normalization_4/Cast_2/ReadVariableOp,^batch_normalization_4/Cast_3/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp%^ecc_conv/FGN_0/MatMul/ReadVariableOp%^ecc_conv/FGN_1/MatMul/ReadVariableOp%^ecc_conv/FGN_2/MatMul/ReadVariableOp(^ecc_conv/FGN_out/BiasAdd/ReadVariableOp'^ecc_conv/FGN_out/MatMul/ReadVariableOp!^ecc_conv/MatMul_1/ReadVariableOp^gcn_conv/MatMul/ReadVariableOp!^gcn_conv_1/MatMul/ReadVariableOp!^gcn_conv_2/MatMul/ReadVariableOp!^gcn_conv_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Є
_input_shapesш
Ы:         :         :         ::         ::::::::::::::::::::::::::::::::::::::::::2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2V
)batch_normalization/Cast_2/ReadVariableOp)batch_normalization/Cast_2/ReadVariableOp2V
)batch_normalization/Cast_3/ReadVariableOp)batch_normalization/Cast_3/ReadVariableOp2V
)batch_normalization_1/Cast/ReadVariableOp)batch_normalization_1/Cast/ReadVariableOp2Z
+batch_normalization_1/Cast_1/ReadVariableOp+batch_normalization_1/Cast_1/ReadVariableOp2Z
+batch_normalization_1/Cast_2/ReadVariableOp+batch_normalization_1/Cast_2/ReadVariableOp2Z
+batch_normalization_1/Cast_3/ReadVariableOp+batch_normalization_1/Cast_3/ReadVariableOp2V
)batch_normalization_2/Cast/ReadVariableOp)batch_normalization_2/Cast/ReadVariableOp2Z
+batch_normalization_2/Cast_1/ReadVariableOp+batch_normalization_2/Cast_1/ReadVariableOp2Z
+batch_normalization_2/Cast_2/ReadVariableOp+batch_normalization_2/Cast_2/ReadVariableOp2Z
+batch_normalization_2/Cast_3/ReadVariableOp+batch_normalization_2/Cast_3/ReadVariableOp2V
)batch_normalization_3/Cast/ReadVariableOp)batch_normalization_3/Cast/ReadVariableOp2Z
+batch_normalization_3/Cast_1/ReadVariableOp+batch_normalization_3/Cast_1/ReadVariableOp2Z
+batch_normalization_3/Cast_2/ReadVariableOp+batch_normalization_3/Cast_2/ReadVariableOp2Z
+batch_normalization_3/Cast_3/ReadVariableOp+batch_normalization_3/Cast_3/ReadVariableOp2V
)batch_normalization_4/Cast/ReadVariableOp)batch_normalization_4/Cast/ReadVariableOp2Z
+batch_normalization_4/Cast_1/ReadVariableOp+batch_normalization_4/Cast_1/ReadVariableOp2Z
+batch_normalization_4/Cast_2/ReadVariableOp+batch_normalization_4/Cast_2/ReadVariableOp2Z
+batch_normalization_4/Cast_3/ReadVariableOp+batch_normalization_4/Cast_3/ReadVariableOp2<
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
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2L
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
:         
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:MI
#
_output_shapes
:         
"
_user_specified_name
inputs/2
р
|
'__inference_dense_4_layer_call_fn_63389

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_615032
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
М
t
J__inference_global_sum_pool_layer_call_and_return_conditional_losses_61234

inputs
inputs_1	
identity{

SegmentSum
SegmentSuminputsinputs_1*
T0*
Tindices0	*(
_output_shapes
:         ђ2

SegmentSumh
IdentityIdentitySegmentSum:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ђ:         :P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs
Ъ	
џ
*__inference_gcn_conv_2_layer_call_fn_63217
inputs_0

inputs	
inputs_1
inputs_2	
unknown
identityѕбStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2unknown*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_gcn_conv_2_layer_call_and_return_conditional_losses_611562
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*S
_input_shapesB
@:         ђ:         :         ::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:         ђ
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
╝
е
5__inference_batch_normalization_2_layer_call_fn_63622

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_606282
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Й
е
5__inference_batch_normalization_4_layer_call_fn_63799

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_609412
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
БB
╠
C__inference_ecc_conv_layer_call_and_return_conditional_losses_61054

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
identityѕбFGN_0/MatMul/ReadVariableOpбFGN_1/MatMul/ReadVariableOpбFGN_2/MatMul/ReadVariableOpбFGN_out/BiasAdd/ReadVariableOpбFGN_out/MatMul/ReadVariableOpбMatMul_1/ReadVariableOpD
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
■        2
strided_slice/stackЂ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЪ
FGN_0/MatMul/ReadVariableOpReadVariableOp$fgn_0_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
FGN_0/MatMul/ReadVariableOpЄ
FGN_0/MatMulMatMulinputs_4#FGN_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
FGN_0/MatMulj

FGN_0/ReluReluFGN_0/MatMul:product:0*
T0*'
_output_shapes
:         @2

FGN_0/ReluЪ
FGN_1/MatMul/ReadVariableOpReadVariableOp$fgn_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
FGN_1/MatMul/ReadVariableOpЌ
FGN_1/MatMulMatMulFGN_0/Relu:activations:0#FGN_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
FGN_1/MatMulj

FGN_1/ReluReluFGN_1/MatMul:product:0*
T0*'
_output_shapes
:         @2

FGN_1/ReluЪ
FGN_2/MatMul/ReadVariableOpReadVariableOp$fgn_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
FGN_2/MatMul/ReadVariableOpЌ
FGN_2/MatMulMatMulFGN_1/Relu:activations:0#FGN_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
FGN_2/MatMulj

FGN_2/ReluReluFGN_2/MatMul:product:0*
T0*'
_output_shapes
:         @2

FGN_2/Reluд
FGN_out/MatMul/ReadVariableOpReadVariableOp&fgn_out_matmul_readvariableop_resource*
_output_shapes
:	@└*
dtype02
FGN_out/MatMul/ReadVariableOpъ
FGN_out/MatMulMatMulFGN_2/Relu:activations:0%FGN_out/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └2
FGN_out/MatMulЦ
FGN_out/BiasAdd/ReadVariableOpReadVariableOp'fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:└*
dtype02 
FGN_out/BiasAdd/ReadVariableOpб
FGN_out/BiasAddBiasAddFGN_out/MatMul:product:0&FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └2
FGN_out/BiasAdds
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       @   2
Reshape/shapeЁ
ReshapeReshapeFGN_out/BiasAdd:output:0Reshape/shape:output:0*
T0*+
_output_shapes
:         @2	
Reshape
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ■   2
strided_slice_1/stackЃ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1Ѓ
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2Ћ
strided_slice_1StridedSliceinputs_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stackЃ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1Ѓ
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2Ћ
strided_slice_2StridedSliceinputs_1strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis»
GatherV2GatherV2inputsstrided_slice_2:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:         2

GatherV2Ѓ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_3/stackЄ
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_3/stack_1Є
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_3/stack_2Б
strided_slice_3StridedSliceGatherV2:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2
strided_slice_3Ѓ
MatMulBatchMatMulV2strided_slice_3:output:0Reshape:output:0*
T0*+
_output_shapes
:         @2
MatMulЃ
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_4/stackЄ
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_4/stack_1Є
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_4/stack_2а
strided_slice_4StridedSliceMatMul:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4╠
UnsortedSegmentSumUnsortedSegmentSumstrided_slice_4:output:0strided_slice_1:output:0strided_slice:output:0*
T0*
Tindices0	*'
_output_shapes
:         @2
UnsortedSegmentSumЊ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulinputsMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2

MatMul_1v
addAddV2UnsortedSegmentSum:output:0MatMul_1:product:0*
T0*'
_output_shapes
:         @2
addO
ReluReluadd:z:0*
T0*'
_output_shapes
:         @2
ReluЏ
IdentityIdentityRelu:activations:0^FGN_0/MatMul/ReadVariableOp^FGN_1/MatMul/ReadVariableOp^FGN_2/MatMul/ReadVariableOp^FGN_out/BiasAdd/ReadVariableOp^FGN_out/MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:         :         :         ::         ::::::2:
FGN_0/MatMul/ReadVariableOpFGN_0/MatMul/ReadVariableOp2:
FGN_1/MatMul/ReadVariableOpFGN_1/MatMul/ReadVariableOp2:
FGN_2/MatMul/ReadVariableOpFGN_2/MatMul/ReadVariableOp2@
FGN_out/BiasAdd/ReadVariableOpFGN_out/BiasAdd/ReadVariableOp2>
FGN_out/MatMul/ReadVariableOpFGN_out/MatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs
ў	
█
B__inference_dense_3_layer_call_and_return_conditional_losses_63361

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddќ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Њ
╩
E__inference_gcn_conv_1_layer_call_and_return_conditional_losses_61129

inputs
inputs_1	
inputs_2
inputs_3	"
matmul_readvariableop_resource
identityѕбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@ђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЯ
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1inputs_2inputs_3MatMul:product:0*
T0*(
_output_shapes
:         ђ21
/SparseTensorDenseMatMul/SparseTensorDenseMatMulѓ
ReluRelu9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:         ђ2
Relu
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:         @:         :         ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
ГB
╬
C__inference_ecc_conv_layer_call_and_return_conditional_losses_63130
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
identityѕбFGN_0/MatMul/ReadVariableOpбFGN_1/MatMul/ReadVariableOpбFGN_2/MatMul/ReadVariableOpбFGN_out/BiasAdd/ReadVariableOpбFGN_out/MatMul/ReadVariableOpбMatMul_1/ReadVariableOpF
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
■        2
strided_slice/stackЂ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЪ
FGN_0/MatMul/ReadVariableOpReadVariableOp$fgn_0_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
FGN_0/MatMul/ReadVariableOpЅ
FGN_0/MatMulMatMul
inputs_2_0#FGN_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
FGN_0/MatMulj

FGN_0/ReluReluFGN_0/MatMul:product:0*
T0*'
_output_shapes
:         @2

FGN_0/ReluЪ
FGN_1/MatMul/ReadVariableOpReadVariableOp$fgn_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
FGN_1/MatMul/ReadVariableOpЌ
FGN_1/MatMulMatMulFGN_0/Relu:activations:0#FGN_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
FGN_1/MatMulj

FGN_1/ReluReluFGN_1/MatMul:product:0*
T0*'
_output_shapes
:         @2

FGN_1/ReluЪ
FGN_2/MatMul/ReadVariableOpReadVariableOp$fgn_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
FGN_2/MatMul/ReadVariableOpЌ
FGN_2/MatMulMatMulFGN_1/Relu:activations:0#FGN_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
FGN_2/MatMulj

FGN_2/ReluReluFGN_2/MatMul:product:0*
T0*'
_output_shapes
:         @2

FGN_2/Reluд
FGN_out/MatMul/ReadVariableOpReadVariableOp&fgn_out_matmul_readvariableop_resource*
_output_shapes
:	@└*
dtype02
FGN_out/MatMul/ReadVariableOpъ
FGN_out/MatMulMatMulFGN_2/Relu:activations:0%FGN_out/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └2
FGN_out/MatMulЦ
FGN_out/BiasAdd/ReadVariableOpReadVariableOp'fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:└*
dtype02 
FGN_out/BiasAdd/ReadVariableOpб
FGN_out/BiasAddBiasAddFGN_out/MatMul:product:0&FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └2
FGN_out/BiasAdds
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       @   2
Reshape/shapeЁ
ReshapeReshapeFGN_out/BiasAdd:output:0Reshape/shape:output:0*
T0*+
_output_shapes
:         @2	
Reshape
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ■   2
strided_slice_1/stackЃ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1Ѓ
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2Њ
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stackЃ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1Ѓ
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2Њ
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis▒
GatherV2GatherV2inputs_0strided_slice_2:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:         2

GatherV2Ѓ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_3/stackЄ
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_3/stack_1Є
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_3/stack_2Б
strided_slice_3StridedSliceGatherV2:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2
strided_slice_3Ѓ
MatMulBatchMatMulV2strided_slice_3:output:0Reshape:output:0*
T0*+
_output_shapes
:         @2
MatMulЃ
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_4/stackЄ
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_4/stack_1Є
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_4/stack_2а
strided_slice_4StridedSliceMatMul:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4╠
UnsortedSegmentSumUnsortedSegmentSumstrided_slice_4:output:0strided_slice_1:output:0strided_slice:output:0*
T0*
Tindices0	*'
_output_shapes
:         @2
UnsortedSegmentSumЊ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulinputs_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2

MatMul_1v
addAddV2UnsortedSegmentSum:output:0MatMul_1:product:0*
T0*'
_output_shapes
:         @2
addO
ReluReluadd:z:0*
T0*'
_output_shapes
:         @2
ReluЏ
IdentityIdentityRelu:activations:0^FGN_0/MatMul/ReadVariableOp^FGN_1/MatMul/ReadVariableOp^FGN_2/MatMul/ReadVariableOp^FGN_out/BiasAdd/ReadVariableOp^FGN_out/MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:         :         :         ::         ::::::2:
FGN_0/MatMul/ReadVariableOpFGN_0/MatMul/ReadVariableOp2:
FGN_1/MatMul/ReadVariableOpFGN_1/MatMul/ReadVariableOp2:
FGN_2/MatMul/ReadVariableOpFGN_2/MatMul/ReadVariableOp2@
FGN_out/BiasAdd/ReadVariableOpFGN_out/BiasAdd/ReadVariableOp2>
FGN_out/MatMul/ReadVariableOpFGN_out/MatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/2
ў	
█
B__inference_dense_1_layer_call_and_return_conditional_losses_61317

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddќ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ь
Ф
%__inference_model_layer_call_fn_63072
inputs_0

inputs	
inputs_1
inputs_2	

inputs_2_0	
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

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identityѕбStatefulPartitionedCall╩
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2
inputs_2_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*:
Tin3
12/			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *L
_read_only_resource_inputs.
,*	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8ѓ *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_621442
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Є
_input_shapesш
Ы:         :         :         ::         ::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:MI
#
_output_shapes
:         
"
_user_specified_name
inputs/2
║
з
(__inference_ecc_conv_layer_call_fn_63151
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
identityѕбStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2
inputs_2_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*(
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *L
fGRE
C__inference_ecc_conv_layer_call_and_return_conditional_losses_610542
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:         :         :         ::         ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/2
Ю	
џ
*__inference_gcn_conv_1_layer_call_fn_63195
inputs_0

inputs	
inputs_1
inputs_2	
unknown
identityѕбStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2unknown*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_gcn_conv_1_layer_call_and_return_conditional_losses_611292
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:         @:         :         ::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         @
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
▒а
§#
@__inference_model_layer_call_and_return_conditional_losses_62646
inputs_0

inputs	
inputs_1
inputs_2	

inputs_2_0	1
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
%dense_biasadd_readvariableop_resource-
)batch_normalization_assignmovingavg_62459/
+batch_normalization_assignmovingavg_1_624654
0batch_normalization_cast_readvariableop_resource6
2batch_normalization_cast_1_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource/
+batch_normalization_1_assignmovingavg_624981
-batch_normalization_1_assignmovingavg_1_625046
2batch_normalization_1_cast_readvariableop_resource8
4batch_normalization_1_cast_1_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource/
+batch_normalization_2_assignmovingavg_625371
-batch_normalization_2_assignmovingavg_1_625436
2batch_normalization_2_cast_readvariableop_resource8
4batch_normalization_2_cast_1_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource/
+batch_normalization_3_assignmovingavg_625761
-batch_normalization_3_assignmovingavg_1_625826
2batch_normalization_3_cast_readvariableop_resource8
4batch_normalization_3_cast_1_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource/
+batch_normalization_4_assignmovingavg_626151
-batch_normalization_4_assignmovingavg_1_626216
2batch_normalization_4_cast_readvariableop_resource8
4batch_normalization_4_cast_1_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identityѕб7batch_normalization/AssignMovingAvg/AssignSubVariableOpб2batch_normalization/AssignMovingAvg/ReadVariableOpб9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpб4batch_normalization/AssignMovingAvg_1/ReadVariableOpб'batch_normalization/Cast/ReadVariableOpб)batch_normalization/Cast_1/ReadVariableOpб9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpб4batch_normalization_1/AssignMovingAvg/ReadVariableOpб;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpб6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpб)batch_normalization_1/Cast/ReadVariableOpб+batch_normalization_1/Cast_1/ReadVariableOpб9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpб4batch_normalization_2/AssignMovingAvg/ReadVariableOpб;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpб6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpб)batch_normalization_2/Cast/ReadVariableOpб+batch_normalization_2/Cast_1/ReadVariableOpб9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpб4batch_normalization_3/AssignMovingAvg/ReadVariableOpб;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpб6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpб)batch_normalization_3/Cast/ReadVariableOpб+batch_normalization_3/Cast_1/ReadVariableOpб9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpб4batch_normalization_4/AssignMovingAvg/ReadVariableOpб;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpб6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpб)batch_normalization_4/Cast/ReadVariableOpб+batch_normalization_4/Cast_1/ReadVariableOpбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpбdense_2/BiasAdd/ReadVariableOpбdense_2/MatMul/ReadVariableOpбdense_3/BiasAdd/ReadVariableOpбdense_3/MatMul/ReadVariableOpбdense_4/BiasAdd/ReadVariableOpбdense_4/MatMul/ReadVariableOpбdense_5/BiasAdd/ReadVariableOpбdense_5/MatMul/ReadVariableOpб$ecc_conv/FGN_0/MatMul/ReadVariableOpб$ecc_conv/FGN_1/MatMul/ReadVariableOpб$ecc_conv/FGN_2/MatMul/ReadVariableOpб'ecc_conv/FGN_out/BiasAdd/ReadVariableOpб&ecc_conv/FGN_out/MatMul/ReadVariableOpб ecc_conv/MatMul_1/ReadVariableOpбgcn_conv/MatMul/ReadVariableOpб gcn_conv_1/MatMul/ReadVariableOpб gcn_conv_2/MatMul/ReadVariableOpб gcn_conv_3/MatMul/ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2Ѕ
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stackЃ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1Ѓ
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2Њ
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis▒
GatherV2GatherV2inputs_0strided_slice_1:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:         2

GatherV2d
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_1/axisх

GatherV2_1GatherV2inputs_0strided_slice:output:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:         2

GatherV2_1k
SubSubGatherV2:output:0GatherV2_1:output:0*
T0*'
_output_shapes
:         2
Sub
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stackЃ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1Ѓ
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2ђ
strided_slice_2StridedSliceSub:z:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
strided_slice_2f
SquareSquarestrided_slice_2:output:0*
T0*'
_output_shapes
:         2
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesk
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
SumP
SqrtSqrtSum:output:0*
T0*#
_output_shapes
:         2
Sqrt
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stackЃ
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1Ѓ
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2ђ
strided_slice_3StridedSliceSub:z:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
strided_slice_3k
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         2
ExpandDims/dim{

ExpandDims
ExpandDimsSqrt:y:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:         2

ExpandDimsЁ

div_no_nanDivNoNanstrided_slice_3:output:0ExpandDims:output:0*
T0*'
_output_shapes
:         2

div_no_nan
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stackЃ
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_4/stack_1Ѓ
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2ђ
strided_slice_4StridedSliceSub:z:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
strided_slice_4o
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
         2
ExpandDims_1/dimЂ
ExpandDims_1
ExpandDimsSqrt:y:0ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:         2
ExpandDims_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis«
concatConcatV2strided_slice_4:output:0ExpandDims_1:output:0div_no_nan:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
concatX
ecc_conv/ShapeShapeinputs_0*
T0*
_output_shapes
:2
ecc_conv/ShapeЈ
ecc_conv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        2
ecc_conv/strided_slice/stackЊ
ecc_conv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         2 
ecc_conv/strided_slice/stack_1і
ecc_conv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
ecc_conv/strided_slice/stack_2ў
ecc_conv/strided_sliceStridedSliceecc_conv/Shape:output:0%ecc_conv/strided_slice/stack:output:0'ecc_conv/strided_slice/stack_1:output:0'ecc_conv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ecc_conv/strided_slice║
$ecc_conv/FGN_0/MatMul/ReadVariableOpReadVariableOp-ecc_conv_fgn_0_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02&
$ecc_conv/FGN_0/MatMul/ReadVariableOpЕ
ecc_conv/FGN_0/MatMulMatMulconcat:output:0,ecc_conv/FGN_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
ecc_conv/FGN_0/MatMulЁ
ecc_conv/FGN_0/ReluReluecc_conv/FGN_0/MatMul:product:0*
T0*'
_output_shapes
:         @2
ecc_conv/FGN_0/Relu║
$ecc_conv/FGN_1/MatMul/ReadVariableOpReadVariableOp-ecc_conv_fgn_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02&
$ecc_conv/FGN_1/MatMul/ReadVariableOp╗
ecc_conv/FGN_1/MatMulMatMul!ecc_conv/FGN_0/Relu:activations:0,ecc_conv/FGN_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
ecc_conv/FGN_1/MatMulЁ
ecc_conv/FGN_1/ReluReluecc_conv/FGN_1/MatMul:product:0*
T0*'
_output_shapes
:         @2
ecc_conv/FGN_1/Relu║
$ecc_conv/FGN_2/MatMul/ReadVariableOpReadVariableOp-ecc_conv_fgn_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02&
$ecc_conv/FGN_2/MatMul/ReadVariableOp╗
ecc_conv/FGN_2/MatMulMatMul!ecc_conv/FGN_1/Relu:activations:0,ecc_conv/FGN_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
ecc_conv/FGN_2/MatMulЁ
ecc_conv/FGN_2/ReluReluecc_conv/FGN_2/MatMul:product:0*
T0*'
_output_shapes
:         @2
ecc_conv/FGN_2/Relu┴
&ecc_conv/FGN_out/MatMul/ReadVariableOpReadVariableOp/ecc_conv_fgn_out_matmul_readvariableop_resource*
_output_shapes
:	@└*
dtype02(
&ecc_conv/FGN_out/MatMul/ReadVariableOp┬
ecc_conv/FGN_out/MatMulMatMul!ecc_conv/FGN_2/Relu:activations:0.ecc_conv/FGN_out/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └2
ecc_conv/FGN_out/MatMul└
'ecc_conv/FGN_out/BiasAdd/ReadVariableOpReadVariableOp0ecc_conv_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:└*
dtype02)
'ecc_conv/FGN_out/BiasAdd/ReadVariableOpк
ecc_conv/FGN_out/BiasAddBiasAdd!ecc_conv/FGN_out/MatMul:product:0/ecc_conv/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └2
ecc_conv/FGN_out/BiasAddЁ
ecc_conv/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       @   2
ecc_conv/Reshape/shapeЕ
ecc_conv/ReshapeReshape!ecc_conv/FGN_out/BiasAdd:output:0ecc_conv/Reshape/shape:output:0*
T0*+
_output_shapes
:         @2
ecc_conv/ReshapeЉ
ecc_conv/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ■   2 
ecc_conv/strided_slice_1/stackЋ
 ecc_conv/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 ecc_conv/strided_slice_1/stack_1Ћ
 ecc_conv/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 ecc_conv/strided_slice_1/stack_2└
ecc_conv/strided_slice_1StridedSliceinputs'ecc_conv/strided_slice_1/stack:output:0)ecc_conv/strided_slice_1/stack_1:output:0)ecc_conv/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
ecc_conv/strided_slice_1Љ
ecc_conv/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
ecc_conv/strided_slice_2/stackЋ
 ecc_conv/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 ecc_conv/strided_slice_2/stack_1Ћ
 ecc_conv/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 ecc_conv/strided_slice_2/stack_2└
ecc_conv/strided_slice_2StridedSliceinputs'ecc_conv/strided_slice_2/stack:output:0)ecc_conv/strided_slice_2/stack_1:output:0)ecc_conv/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
ecc_conv/strided_slice_2r
ecc_conv/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
ecc_conv/GatherV2/axisН
ecc_conv/GatherV2GatherV2inputs_0!ecc_conv/strided_slice_2:output:0ecc_conv/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:         2
ecc_conv/GatherV2Ћ
ecc_conv/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2 
ecc_conv/strided_slice_3/stackЎ
 ecc_conv/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2"
 ecc_conv/strided_slice_3/stack_1Ў
 ecc_conv/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2"
 ecc_conv/strided_slice_3/stack_2┘
ecc_conv/strided_slice_3StridedSliceecc_conv/GatherV2:output:0'ecc_conv/strided_slice_3/stack:output:0)ecc_conv/strided_slice_3/stack_1:output:0)ecc_conv/strided_slice_3/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2
ecc_conv/strided_slice_3Д
ecc_conv/MatMulBatchMatMulV2!ecc_conv/strided_slice_3:output:0ecc_conv/Reshape:output:0*
T0*+
_output_shapes
:         @2
ecc_conv/MatMulЋ
ecc_conv/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2 
ecc_conv/strided_slice_4/stackЎ
 ecc_conv/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2"
 ecc_conv/strided_slice_4/stack_1Ў
 ecc_conv/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2"
 ecc_conv/strided_slice_4/stack_2о
ecc_conv/strided_slice_4StridedSliceecc_conv/MatMul:output:0'ecc_conv/strided_slice_4/stack:output:0)ecc_conv/strided_slice_4/stack_1:output:0)ecc_conv/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*

begin_mask*
end_mask*
shrink_axis_mask2
ecc_conv/strided_slice_4щ
ecc_conv/UnsortedSegmentSumUnsortedSegmentSum!ecc_conv/strided_slice_4:output:0!ecc_conv/strided_slice_1:output:0ecc_conv/strided_slice:output:0*
T0*
Tindices0	*'
_output_shapes
:         @2
ecc_conv/UnsortedSegmentSum«
 ecc_conv/MatMul_1/ReadVariableOpReadVariableOp)ecc_conv_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02"
 ecc_conv/MatMul_1/ReadVariableOpќ
ecc_conv/MatMul_1MatMulinputs_0(ecc_conv/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
ecc_conv/MatMul_1џ
ecc_conv/addAddV2$ecc_conv/UnsortedSegmentSum:output:0ecc_conv/MatMul_1:product:0*
T0*'
_output_shapes
:         @2
ecc_conv/addj
ecc_conv/ReluReluecc_conv/add:z:0*
T0*'
_output_shapes
:         @2
ecc_conv/Reluе
gcn_conv/MatMul/ReadVariableOpReadVariableOp'gcn_conv_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
gcn_conv/MatMul/ReadVariableOpБ
gcn_conv/MatMulMatMulecc_conv/Relu:activations:0&gcn_conv/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
gcn_conv/MatMulЭ
8gcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2gcn_conv/MatMul:product:0*
T0*'
_output_shapes
:         @2:
8gcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMulю
gcn_conv/ReluReluBgcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*'
_output_shapes
:         @2
gcn_conv/Relu»
 gcn_conv_1/MatMul/ReadVariableOpReadVariableOp)gcn_conv_1_matmul_readvariableop_resource*
_output_shapes
:	@ђ*
dtype02"
 gcn_conv_1/MatMul/ReadVariableOpф
gcn_conv_1/MatMulMatMulgcn_conv/Relu:activations:0(gcn_conv_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
gcn_conv_1/MatMul 
:gcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2gcn_conv_1/MatMul:product:0*
T0*(
_output_shapes
:         ђ2<
:gcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMulБ
gcn_conv_1/ReluReluDgcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:         ђ2
gcn_conv_1/Relu░
 gcn_conv_2/MatMul/ReadVariableOpReadVariableOp)gcn_conv_2_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02"
 gcn_conv_2/MatMul/ReadVariableOpг
gcn_conv_2/MatMulMatMulgcn_conv_1/Relu:activations:0(gcn_conv_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
gcn_conv_2/MatMul 
:gcn_conv_2/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2gcn_conv_2/MatMul:product:0*
T0*(
_output_shapes
:         ђ2<
:gcn_conv_2/SparseTensorDenseMatMul/SparseTensorDenseMatMulБ
gcn_conv_2/ReluReluDgcn_conv_2/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:         ђ2
gcn_conv_2/Relu░
 gcn_conv_3/MatMul/ReadVariableOpReadVariableOp)gcn_conv_3_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02"
 gcn_conv_3/MatMul/ReadVariableOpг
gcn_conv_3/MatMulMatMulgcn_conv_2/Relu:activations:0(gcn_conv_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
gcn_conv_3/MatMul 
:gcn_conv_3/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2gcn_conv_3/MatMul:product:0*
T0*(
_output_shapes
:         ђ2<
:gcn_conv_3/SparseTensorDenseMatMul/SparseTensorDenseMatMulБ
gcn_conv_3/ReluReluDgcn_conv_3/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:         ђ2
gcn_conv_3/Relu┤
global_max_pool/SegmentMax
SegmentMaxgcn_conv_3/Relu:activations:0
inputs_2_0*
T0*
Tindices0	*(
_output_shapes
:         ђ2
global_max_pool/SegmentMaxи
global_avg_pool/SegmentMeanSegmentMeangcn_conv_3/Relu:activations:0
inputs_2_0*
T0*
Tindices0	*(
_output_shapes
:         ђ2
global_avg_pool/SegmentMean┤
global_sum_pool/SegmentSum
SegmentSumgcn_conv_3/Relu:activations:0
inputs_2_0*
T0*
Tindices0	*(
_output_shapes
:         ђ2
global_sum_pool/SegmentSum`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axisС
concat_1ConcatV2#global_max_pool/SegmentMax:output:0$global_avg_pool/SegmentMean:output:0#global_sum_pool/SegmentSum:output:0concat_1/axis:output:0*
N*
T0*(
_output_shapes
:         ђ2

concat_1А
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
dense/MatMul/ReadVariableOpЉ
dense/MatMulMatMulconcat_1:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense/MatMulЪ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
dense/BiasAdd/ReadVariableOpџ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense/BiasAddЇ
leaky_re_lu/LeakyRelu	LeakyReludense/BiasAdd:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
leaky_re_lu/LeakyRelu▓
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 24
2batch_normalization/moments/mean/reduction_indicesж
 batch_normalization/moments/meanMean#leaky_re_lu/LeakyRelu:activations:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2"
 batch_normalization/moments/mean╣
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	ђ2*
(batch_normalization/moments/StopGradient■
-batch_normalization/moments/SquaredDifferenceSquaredDifference#leaky_re_lu/LeakyRelu:activations:01batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:         ђ2/
-batch_normalization/moments/SquaredDifference║
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization/moments/variance/reduction_indicesЃ
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2&
$batch_normalization/moments/varianceй
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2%
#batch_normalization/moments/Squeeze┼
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1Є
)batch_normalization/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/62459*
_output_shapes
: *
dtype0*
valueB
 *
О#<2+
)batch_normalization/AssignMovingAvg/decay¤
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp)batch_normalization_assignmovingavg_62459*
_output_shapes	
:ђ*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOpН
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/62459*
_output_shapes	
:ђ2)
'batch_normalization/AssignMovingAvg/sub╠
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/62459*
_output_shapes	
:ђ2)
'batch_normalization/AssignMovingAvg/mulЦ
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp)batch_normalization_assignmovingavg_62459+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/62459*
_output_shapes
 *
dtype029
7batch_normalization/AssignMovingAvg/AssignSubVariableOpЇ
+batch_normalization/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/62465*
_output_shapes
: *
dtype0*
valueB
 *
О#<2-
+batch_normalization/AssignMovingAvg_1/decayН
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp+batch_normalization_assignmovingavg_1_62465*
_output_shapes	
:ђ*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp▀
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/62465*
_output_shapes	
:ђ2+
)batch_normalization/AssignMovingAvg_1/subо
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/62465*
_output_shapes	
:ђ2+
)batch_normalization/AssignMovingAvg_1/mul▒
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp+batch_normalization_assignmovingavg_1_62465-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/62465*
_output_shapes
 *
dtype02;
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp└
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02)
'batch_normalization/Cast/ReadVariableOpк
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)batch_normalization/Cast_1/ReadVariableOpЈ
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2%
#batch_normalization/batchnorm/add/yМ
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2#
!batch_normalization/batchnorm/addа
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization/batchnorm/Rsqrt¤
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2#
!batch_normalization/batchnorm/mulл
#batch_normalization/batchnorm/mul_1Mul#leaky_re_lu/LeakyRelu:activations:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2%
#batch_normalization/batchnorm/mul_1╠
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization/batchnorm/mul_2═
!batch_normalization/batchnorm/subSub/batch_normalization/Cast/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2#
!batch_normalization/batchnorm/subо
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2%
#batch_normalization/batchnorm/add_1Д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
dense_1/MatMul/ReadVariableOpГ
dense_1/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_1/MatMulЦ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_1/BiasAddЊ
leaky_re_lu/LeakyRelu_1	LeakyReludense_1/BiasAdd:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
leaky_re_lu/LeakyRelu_1Х
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_1/moments/mean/reduction_indicesы
"batch_normalization_1/moments/meanMean%leaky_re_lu/LeakyRelu_1:activations:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2$
"batch_normalization_1/moments/mean┐
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	ђ2,
*batch_normalization_1/moments/StopGradientє
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference%leaky_re_lu/LeakyRelu_1:activations:03batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:         ђ21
/batch_normalization_1/moments/SquaredDifferenceЙ
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_1/moments/variance/reduction_indicesІ
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2(
&batch_normalization_1/moments/variance├
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze╦
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1Ї
+batch_normalization_1/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*>
_class4
20loc:@batch_normalization_1/AssignMovingAvg/62498*
_output_shapes
: *
dtype0*
valueB
 *
О#<2-
+batch_normalization_1/AssignMovingAvg/decayН
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_1_assignmovingavg_62498*
_output_shapes	
:ђ*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp▀
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*>
_class4
20loc:@batch_normalization_1/AssignMovingAvg/62498*
_output_shapes	
:ђ2+
)batch_normalization_1/AssignMovingAvg/subо
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*>
_class4
20loc:@batch_normalization_1/AssignMovingAvg/62498*
_output_shapes	
:ђ2+
)batch_normalization_1/AssignMovingAvg/mul▒
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_1_assignmovingavg_62498-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*>
_class4
20loc:@batch_normalization_1/AssignMovingAvg/62498*
_output_shapes
 *
dtype02;
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpЊ
-batch_normalization_1/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg_1/62504*
_output_shapes
: *
dtype0*
valueB
 *
О#<2/
-batch_normalization_1/AssignMovingAvg_1/decay█
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_1_assignmovingavg_1_62504*
_output_shapes	
:ђ*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpж
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg_1/62504*
_output_shapes	
:ђ2-
+batch_normalization_1/AssignMovingAvg_1/subЯ
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg_1/62504*
_output_shapes	
:ђ2-
+batch_normalization_1/AssignMovingAvg_1/mulй
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_1_assignmovingavg_1_62504/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg_1/62504*
_output_shapes
 *
dtype02=
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpк
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)batch_normalization_1/Cast/ReadVariableOp╠
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+batch_normalization_1/Cast_1/ReadVariableOpЊ
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_1/batchnorm/add/y█
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_1/batchnorm/addд
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_1/batchnorm/RsqrtО
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_1/batchnorm/mulп
%batch_normalization_1/batchnorm/mul_1Mul%leaky_re_lu/LeakyRelu_1:activations:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2'
%batch_normalization_1/batchnorm/mul_1н
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_1/batchnorm/mul_2Н
#batch_normalization_1/batchnorm/subSub1batch_normalization_1/Cast/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_1/batchnorm/subя
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2'
%batch_normalization_1/batchnorm/add_1Д
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
dense_2/MatMul/ReadVariableOp»
dense_2/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_2/MatMulЦ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
dense_2/BiasAdd/ReadVariableOpб
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_2/BiasAddЊ
leaky_re_lu/LeakyRelu_2	LeakyReludense_2/BiasAdd:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
leaky_re_lu/LeakyRelu_2Х
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_2/moments/mean/reduction_indicesы
"batch_normalization_2/moments/meanMean%leaky_re_lu/LeakyRelu_2:activations:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2$
"batch_normalization_2/moments/mean┐
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:	ђ2,
*batch_normalization_2/moments/StopGradientє
/batch_normalization_2/moments/SquaredDifferenceSquaredDifference%leaky_re_lu/LeakyRelu_2:activations:03batch_normalization_2/moments/StopGradient:output:0*
T0*(
_output_shapes
:         ђ21
/batch_normalization_2/moments/SquaredDifferenceЙ
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_2/moments/variance/reduction_indicesІ
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2(
&batch_normalization_2/moments/variance├
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze╦
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2)
'batch_normalization_2/moments/Squeeze_1Ї
+batch_normalization_2/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*>
_class4
20loc:@batch_normalization_2/AssignMovingAvg/62537*
_output_shapes
: *
dtype0*
valueB
 *
О#<2-
+batch_normalization_2/AssignMovingAvg/decayН
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_2_assignmovingavg_62537*
_output_shapes	
:ђ*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp▀
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*>
_class4
20loc:@batch_normalization_2/AssignMovingAvg/62537*
_output_shapes	
:ђ2+
)batch_normalization_2/AssignMovingAvg/subо
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*>
_class4
20loc:@batch_normalization_2/AssignMovingAvg/62537*
_output_shapes	
:ђ2+
)batch_normalization_2/AssignMovingAvg/mul▒
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_2_assignmovingavg_62537-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*>
_class4
20loc:@batch_normalization_2/AssignMovingAvg/62537*
_output_shapes
 *
dtype02;
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpЊ
-batch_normalization_2/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg_1/62543*
_output_shapes
: *
dtype0*
valueB
 *
О#<2/
-batch_normalization_2/AssignMovingAvg_1/decay█
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_2_assignmovingavg_1_62543*
_output_shapes	
:ђ*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpж
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg_1/62543*
_output_shapes	
:ђ2-
+batch_normalization_2/AssignMovingAvg_1/subЯ
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg_1/62543*
_output_shapes	
:ђ2-
+batch_normalization_2/AssignMovingAvg_1/mulй
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_2_assignmovingavg_1_62543/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg_1/62543*
_output_shapes
 *
dtype02=
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpк
)batch_normalization_2/Cast/ReadVariableOpReadVariableOp2batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)batch_normalization_2/Cast/ReadVariableOp╠
+batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+batch_normalization_2/Cast_1/ReadVariableOpЊ
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_2/batchnorm/add/y█
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_2/batchnorm/addд
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_2/batchnorm/RsqrtО
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:03batch_normalization_2/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_2/batchnorm/mulп
%batch_normalization_2/batchnorm/mul_1Mul%leaky_re_lu/LeakyRelu_2:activations:0'batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2'
%batch_normalization_2/batchnorm/mul_1н
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_2/batchnorm/mul_2Н
#batch_normalization_2/batchnorm/subSub1batch_normalization_2/Cast/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_2/batchnorm/subя
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2'
%batch_normalization_2/batchnorm/add_1Д
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
dense_3/MatMul/ReadVariableOp»
dense_3/MatMulMatMul)batch_normalization_2/batchnorm/add_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_3/MatMulЦ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
dense_3/BiasAdd/ReadVariableOpб
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_3/BiasAddЊ
leaky_re_lu/LeakyRelu_3	LeakyReludense_3/BiasAdd:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
leaky_re_lu/LeakyRelu_3Х
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_3/moments/mean/reduction_indicesы
"batch_normalization_3/moments/meanMean%leaky_re_lu/LeakyRelu_3:activations:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2$
"batch_normalization_3/moments/mean┐
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:	ђ2,
*batch_normalization_3/moments/StopGradientє
/batch_normalization_3/moments/SquaredDifferenceSquaredDifference%leaky_re_lu/LeakyRelu_3:activations:03batch_normalization_3/moments/StopGradient:output:0*
T0*(
_output_shapes
:         ђ21
/batch_normalization_3/moments/SquaredDifferenceЙ
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_3/moments/variance/reduction_indicesІ
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2(
&batch_normalization_3/moments/variance├
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2'
%batch_normalization_3/moments/Squeeze╦
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2)
'batch_normalization_3/moments/Squeeze_1Ї
+batch_normalization_3/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/62576*
_output_shapes
: *
dtype0*
valueB
 *
О#<2-
+batch_normalization_3/AssignMovingAvg/decayН
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_3_assignmovingavg_62576*
_output_shapes	
:ђ*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOp▀
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/62576*
_output_shapes	
:ђ2+
)batch_normalization_3/AssignMovingAvg/subо
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/62576*
_output_shapes	
:ђ2+
)batch_normalization_3/AssignMovingAvg/mul▒
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_3_assignmovingavg_62576-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/62576*
_output_shapes
 *
dtype02;
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpЊ
-batch_normalization_3/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/62582*
_output_shapes
: *
dtype0*
valueB
 *
О#<2/
-batch_normalization_3/AssignMovingAvg_1/decay█
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_3_assignmovingavg_1_62582*
_output_shapes	
:ђ*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpж
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/62582*
_output_shapes	
:ђ2-
+batch_normalization_3/AssignMovingAvg_1/subЯ
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/62582*
_output_shapes	
:ђ2-
+batch_normalization_3/AssignMovingAvg_1/mulй
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_3_assignmovingavg_1_62582/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/62582*
_output_shapes
 *
dtype02=
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpк
)batch_normalization_3/Cast/ReadVariableOpReadVariableOp2batch_normalization_3_cast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)batch_normalization_3/Cast/ReadVariableOp╠
+batch_normalization_3/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_3_cast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+batch_normalization_3/Cast_1/ReadVariableOpЊ
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_3/batchnorm/add/y█
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_3/batchnorm/addд
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_3/batchnorm/RsqrtО
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:03batch_normalization_3/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_3/batchnorm/mulп
%batch_normalization_3/batchnorm/mul_1Mul%leaky_re_lu/LeakyRelu_3:activations:0'batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2'
%batch_normalization_3/batchnorm/mul_1н
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_3/batchnorm/mul_2Н
#batch_normalization_3/batchnorm/subSub1batch_normalization_3/Cast/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_3/batchnorm/subя
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2'
%batch_normalization_3/batchnorm/add_1Д
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
dense_4/MatMul/ReadVariableOp»
dense_4/MatMulMatMul)batch_normalization_3/batchnorm/add_1:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_4/MatMulЦ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
dense_4/BiasAdd/ReadVariableOpб
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_4/BiasAddЊ
leaky_re_lu/LeakyRelu_4	LeakyReludense_4/BiasAdd:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
leaky_re_lu/LeakyRelu_4Х
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_4/moments/mean/reduction_indicesы
"batch_normalization_4/moments/meanMean%leaky_re_lu/LeakyRelu_4:activations:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2$
"batch_normalization_4/moments/mean┐
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:	ђ2,
*batch_normalization_4/moments/StopGradientє
/batch_normalization_4/moments/SquaredDifferenceSquaredDifference%leaky_re_lu/LeakyRelu_4:activations:03batch_normalization_4/moments/StopGradient:output:0*
T0*(
_output_shapes
:         ђ21
/batch_normalization_4/moments/SquaredDifferenceЙ
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_4/moments/variance/reduction_indicesІ
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2(
&batch_normalization_4/moments/variance├
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2'
%batch_normalization_4/moments/Squeeze╦
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2)
'batch_normalization_4/moments/Squeeze_1Ї
+batch_normalization_4/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/62615*
_output_shapes
: *
dtype0*
valueB
 *
О#<2-
+batch_normalization_4/AssignMovingAvg/decayН
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_4_assignmovingavg_62615*
_output_shapes	
:ђ*
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOp▀
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/62615*
_output_shapes	
:ђ2+
)batch_normalization_4/AssignMovingAvg/subо
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/62615*
_output_shapes	
:ђ2+
)batch_normalization_4/AssignMovingAvg/mul▒
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_4_assignmovingavg_62615-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/62615*
_output_shapes
 *
dtype02;
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpЊ
-batch_normalization_4/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/62621*
_output_shapes
: *
dtype0*
valueB
 *
О#<2/
-batch_normalization_4/AssignMovingAvg_1/decay█
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_4_assignmovingavg_1_62621*
_output_shapes	
:ђ*
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpж
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/62621*
_output_shapes	
:ђ2-
+batch_normalization_4/AssignMovingAvg_1/subЯ
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/62621*
_output_shapes	
:ђ2-
+batch_normalization_4/AssignMovingAvg_1/mulй
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_4_assignmovingavg_1_62621/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/62621*
_output_shapes
 *
dtype02=
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpк
)batch_normalization_4/Cast/ReadVariableOpReadVariableOp2batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)batch_normalization_4/Cast/ReadVariableOp╠
+batch_normalization_4/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+batch_normalization_4/Cast_1/ReadVariableOpЊ
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2'
%batch_normalization_4/batchnorm/add/y█
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_4/batchnorm/addд
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_4/batchnorm/RsqrtО
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:03batch_normalization_4/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_4/batchnorm/mulп
%batch_normalization_4/batchnorm/mul_1Mul%leaky_re_lu/LeakyRelu_4:activations:0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2'
%batch_normalization_4/batchnorm/mul_1н
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2'
%batch_normalization_4/batchnorm/mul_2Н
#batch_normalization_4/batchnorm/subSub1batch_normalization_4/Cast/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2%
#batch_normalization_4/batchnorm/subя
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2'
%batch_normalization_4/batchnorm/add_1д
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
dense_5/MatMul/ReadVariableOp«
dense_5/MatMulMatMul)batch_normalization_4/batchnorm/add_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/MatMulц
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOpА
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/BiasAddФ
IdentityIdentitydense_5/BiasAdd:output:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp3^batch_normalization/AssignMovingAvg/ReadVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp5^batch_normalization/AssignMovingAvg_1/ReadVariableOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_1/AssignMovingAvg/ReadVariableOp<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_2/AssignMovingAvg/ReadVariableOp<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_2/Cast/ReadVariableOp,^batch_normalization_2/Cast_1/ReadVariableOp:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_3/AssignMovingAvg/ReadVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_3/Cast/ReadVariableOp,^batch_normalization_3/Cast_1/ReadVariableOp:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_4/AssignMovingAvg/ReadVariableOp<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_4/Cast/ReadVariableOp,^batch_normalization_4/Cast_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp%^ecc_conv/FGN_0/MatMul/ReadVariableOp%^ecc_conv/FGN_1/MatMul/ReadVariableOp%^ecc_conv/FGN_2/MatMul/ReadVariableOp(^ecc_conv/FGN_out/BiasAdd/ReadVariableOp'^ecc_conv/FGN_out/MatMul/ReadVariableOp!^ecc_conv/MatMul_1/ReadVariableOp^gcn_conv/MatMul/ReadVariableOp!^gcn_conv_1/MatMul/ReadVariableOp!^gcn_conv_2/MatMul/ReadVariableOp!^gcn_conv_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Є
_input_shapesш
Ы:         :         :         ::         ::::::::::::::::::::::::::::::::::::::::::2r
7batch_normalization/AssignMovingAvg/AssignSubVariableOp7batch_normalization/AssignMovingAvg/AssignSubVariableOp2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2v
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp2l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2v
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_1/Cast/ReadVariableOp)batch_normalization_1/Cast/ReadVariableOp2Z
+batch_normalization_1/Cast_1/ReadVariableOp+batch_normalization_1/Cast_1/ReadVariableOp2v
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_2/Cast/ReadVariableOp)batch_normalization_2/Cast/ReadVariableOp2Z
+batch_normalization_2/Cast_1/ReadVariableOp+batch_normalization_2/Cast_1/ReadVariableOp2v
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_3/Cast/ReadVariableOp)batch_normalization_3/Cast/ReadVariableOp2Z
+batch_normalization_3/Cast_1/ReadVariableOp+batch_normalization_3/Cast_1/ReadVariableOp2v
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_4/Cast/ReadVariableOp)batch_normalization_4/Cast/ReadVariableOp2Z
+batch_normalization_4/Cast_1/ReadVariableOp+batch_normalization_4/Cast_1/ReadVariableOp2<
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
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2L
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
:         
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:MI
#
_output_shapes
:         
"
_user_specified_name
inputs/2
б/
Г
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_60768

inputs
assignmovingavg_60743
assignmovingavg_1_60749 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб AssignMovingAvg_1/ReadVariableOpбCast/ReadVariableOpбCast_1/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesљ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ђ2
moments/StopGradientЦ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         ђ2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeeze_1╦
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/60743*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayЊ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_60743*
_output_shapes	
:ђ*
dtype02 
AssignMovingAvg/ReadVariableOpы
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/60743*
_output_shapes	
:ђ2
AssignMovingAvg/subУ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/60743*
_output_shapes	
:ђ2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_60743AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/60743*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЛ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/60749*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayЎ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_60749*
_output_shapes	
:ђ*
dtype02"
 AssignMovingAvg_1/ReadVariableOpч
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/60749*
_output_shapes	
:ђ2
AssignMovingAvg_1/subЫ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/60749*
_output_shapes	
:ђ2
AssignMovingAvg_1/mul╣
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_60749AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/60749*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/add_1е
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ў	
█
B__inference_dense_1_layer_call_and_return_conditional_losses_63323

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddќ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╦
Д
#__inference_signature_wrapper_62326

args_0
args_0_1	
args_0_2
args_0_3	
args_0_4	
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

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identityѕбStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallargs_0args_0_1args_0_2args_0_3args_0_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*:
Tin3
12/			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *L
_read_only_resource_inputs.
,*	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8ѓ *)
f$R"
 __inference__wrapped_model_602522
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Є
_input_shapesш
Ы:         :         :         ::         ::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameargs_0:QM
'
_output_shapes
:         
"
_user_specified_name
args_0_1:MI
#
_output_shapes
:         
"
_user_specified_name
args_0_2:D@

_output_shapes
:
"
_user_specified_name
args_0_3:MI
#
_output_shapes
:         
"
_user_specified_name
args_0_4
ў
█
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_63527

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identityѕбCast/ReadVariableOpбCast_1/ReadVariableOpбCast_2/ReadVariableOpбCast_3/ReadVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_1/ReadVariableOpі
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_2/ReadVariableOpі
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yє
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/add_1к
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ќ	
┘
@__inference_dense_layer_call_and_return_conditional_losses_63304

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddќ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ц
[
/__inference_global_sum_pool_layer_call_fn_63275
inputs_0
inputs_1	
identity█
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_global_sum_pool_layer_call_and_return_conditional_losses_612342
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ђ:         :R N
(
_output_shapes
:         ђ
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:         
"
_user_specified_name
inputs/1
б/
Г
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_60628

inputs
assignmovingavg_60603
assignmovingavg_1_60609 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб AssignMovingAvg_1/ReadVariableOpбCast/ReadVariableOpбCast_1/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesљ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ђ2
moments/StopGradientЦ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         ђ2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeeze_1╦
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/60603*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayЊ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_60603*
_output_shapes	
:ђ*
dtype02 
AssignMovingAvg/ReadVariableOpы
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/60603*
_output_shapes	
:ђ2
AssignMovingAvg/subУ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/60603*
_output_shapes	
:ђ2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_60603AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/60603*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЛ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/60609*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayЎ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_60609*
_output_shapes	
:ђ*
dtype02"
 AssignMovingAvg_1/ReadVariableOpч
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/60609*
_output_shapes	
:ђ2
AssignMovingAvg_1/subЫ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/60609*
_output_shapes	
:ђ2
AssignMovingAvg_1/mul╣
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_60609AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/60609*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/add_1е
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
█
v
J__inference_global_sum_pool_layer_call_and_return_conditional_losses_63269
inputs_0
inputs_1	
identity}

SegmentSum
SegmentSuminputs_0inputs_1*
T0*
Tindices0	*(
_output_shapes
:         ђ2

SegmentSumh
IdentityIdentitySegmentSum:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ђ:         :R N
(
_output_shapes
:         ђ
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:         
"
_user_specified_name
inputs/1
Ъ	
џ
*__inference_gcn_conv_3_layer_call_fn_63239
inputs_0

inputs	
inputs_1
inputs_2	
unknown
identityѕбStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2unknown*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_gcn_conv_3_layer_call_and_return_conditional_losses_611832
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*S
_input_shapesB
@:         ђ:         :         ::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:         ђ
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
Њ	
█
B__inference_dense_5_layer_call_and_return_conditional_losses_63285

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddЋ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ќ	
ў
(__inference_gcn_conv_layer_call_fn_63173
inputs_0

inputs	
inputs_1
inputs_2	
unknown
identityѕбStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2unknown*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *L
fGRE
C__inference_gcn_conv_layer_call_and_return_conditional_losses_611022
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:         @:         :         ::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         @
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
ц
[
/__inference_global_avg_pool_layer_call_fn_63263
inputs_0
inputs_1	
identity█
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_global_avg_pool_layer_call_and_return_conditional_losses_612192
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ђ:         :R N
(
_output_shapes
:         ђ
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:         
"
_user_specified_name
inputs/1
▀
|
'__inference_dense_5_layer_call_fn_63294

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_615652
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
И
д
3__inference_batch_normalization_layer_call_fn_63458

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_603482
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
║
д
3__inference_batch_normalization_layer_call_fn_63471

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_603812
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
р
|
'__inference_dense_2_layer_call_fn_63351

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_613792
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
р
|
'__inference_dense_1_layer_call_fn_63332

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_613172
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╝
е
5__inference_batch_normalization_1_layer_call_fn_63540

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_604882
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
р
|
'__inference_dense_3_layer_call_fn_63370

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_614412
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ў
█
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_63609

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identityѕбCast/ReadVariableOpбCast_1/ReadVariableOpбCast_2/ReadVariableOpбCast_3/ReadVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_1/ReadVariableOpі
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_2/ReadVariableOpі
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yє
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/add_1к
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ЫX
Ѕ
__inference__traced_save_63952
file_prefix9
5savev2_model_ecc_conv_root_kernel_read_readvariableop4
0savev2_model_gcn_conv_kernel_read_readvariableop6
2savev2_model_gcn_conv_1_kernel_read_readvariableop6
2savev2_model_gcn_conv_2_kernel_read_readvariableop6
2savev2_model_gcn_conv_3_kernel_read_readvariableop3
/savev2_model_dense_5_kernel_read_readvariableop1
-savev2_model_dense_5_bias_read_readvariableop:
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
-savev2_model_dense_4_bias_read_readvariableop>
:savev2_model_batch_normalization_gamma_read_readvariableop=
9savev2_model_batch_normalization_beta_read_readvariableop@
<savev2_model_batch_normalization_1_gamma_read_readvariableop?
;savev2_model_batch_normalization_1_beta_read_readvariableop@
<savev2_model_batch_normalization_2_gamma_read_readvariableop?
;savev2_model_batch_normalization_2_beta_read_readvariableop@
<savev2_model_batch_normalization_3_gamma_read_readvariableop?
;savev2_model_batch_normalization_3_beta_read_readvariableop@
<savev2_model_batch_normalization_4_gamma_read_readvariableop?
;savev2_model_batch_normalization_4_beta_read_readvariableopD
@savev2_model_batch_normalization_moving_mean_read_readvariableopH
Dsavev2_model_batch_normalization_moving_variance_read_readvariableopF
Bsavev2_model_batch_normalization_1_moving_mean_read_readvariableopJ
Fsavev2_model_batch_normalization_1_moving_variance_read_readvariableopF
Bsavev2_model_batch_normalization_2_moving_mean_read_readvariableopJ
Fsavev2_model_batch_normalization_2_moving_variance_read_readvariableopF
Bsavev2_model_batch_normalization_3_moving_mean_read_readvariableopJ
Fsavev2_model_batch_normalization_3_moving_variance_read_readvariableopF
Bsavev2_model_batch_normalization_4_moving_mean_read_readvariableopJ
Fsavev2_model_batch_normalization_4_moving_variance_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
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
Const_1І
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
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename╚
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*┌
valueлB═+B+ECC1/root_kernel/.ATTRIBUTES/VARIABLE_VALUEB&GCN1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&GCN2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&GCN3/kernel/.ATTRIBUTES/VARIABLE_VALUEB&GCN4/kernel/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesя
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices┌
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:05savev2_model_ecc_conv_root_kernel_read_readvariableop0savev2_model_gcn_conv_kernel_read_readvariableop2savev2_model_gcn_conv_1_kernel_read_readvariableop2savev2_model_gcn_conv_2_kernel_read_readvariableop2savev2_model_gcn_conv_3_kernel_read_readvariableop/savev2_model_dense_5_kernel_read_readvariableop-savev2_model_dense_5_bias_read_readvariableop6savev2_model_ecc_conv_fgn_0_kernel_read_readvariableop6savev2_model_ecc_conv_fgn_1_kernel_read_readvariableop6savev2_model_ecc_conv_fgn_2_kernel_read_readvariableop8savev2_model_ecc_conv_fgn_out_kernel_read_readvariableop6savev2_model_ecc_conv_fgn_out_bias_read_readvariableop-savev2_model_dense_kernel_read_readvariableop+savev2_model_dense_bias_read_readvariableop/savev2_model_dense_1_kernel_read_readvariableop-savev2_model_dense_1_bias_read_readvariableop/savev2_model_dense_2_kernel_read_readvariableop-savev2_model_dense_2_bias_read_readvariableop/savev2_model_dense_3_kernel_read_readvariableop-savev2_model_dense_3_bias_read_readvariableop/savev2_model_dense_4_kernel_read_readvariableop-savev2_model_dense_4_bias_read_readvariableop:savev2_model_batch_normalization_gamma_read_readvariableop9savev2_model_batch_normalization_beta_read_readvariableop<savev2_model_batch_normalization_1_gamma_read_readvariableop;savev2_model_batch_normalization_1_beta_read_readvariableop<savev2_model_batch_normalization_2_gamma_read_readvariableop;savev2_model_batch_normalization_2_beta_read_readvariableop<savev2_model_batch_normalization_3_gamma_read_readvariableop;savev2_model_batch_normalization_3_beta_read_readvariableop<savev2_model_batch_normalization_4_gamma_read_readvariableop;savev2_model_batch_normalization_4_beta_read_readvariableop@savev2_model_batch_normalization_moving_mean_read_readvariableopDsavev2_model_batch_normalization_moving_variance_read_readvariableopBsavev2_model_batch_normalization_1_moving_mean_read_readvariableopFsavev2_model_batch_normalization_1_moving_variance_read_readvariableopBsavev2_model_batch_normalization_2_moving_mean_read_readvariableopFsavev2_model_batch_normalization_2_moving_variance_read_readvariableopBsavev2_model_batch_normalization_3_moving_mean_read_readvariableopFsavev2_model_batch_normalization_3_moving_variance_read_readvariableopBsavev2_model_batch_normalization_4_moving_mean_read_readvariableopFsavev2_model_batch_normalization_4_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *9
dtypes/
-2+2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
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

identity_1Identity_1:output:0*Ч
_input_shapesЖ
у: :@:@@:	@ђ:
ђђ:
ђђ:	ђ::@:@@:@@:	@└:└:
ђђ:ђ:
ђђ:ђ:
ђђ:ђ:
ђђ:ђ:
ђђ:ђ:ђ:ђ:ђ:ђ:ђ:ђ:ђ:ђ:ђ:ђ:ђ:ђ:ђ:ђ:ђ:ђ:ђ:ђ:ђ:ђ: 2(
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
:	@ђ:&"
 
_output_shapes
:
ђђ:&"
 
_output_shapes
:
ђђ:%!

_output_shapes
:	ђ: 

_output_shapes
::$ 

_output_shapes

:@:$	 

_output_shapes

:@@:$
 

_output_shapes

:@@:%!

_output_shapes
:	@└:!

_output_shapes	
:└:&"
 
_output_shapes
:
ђђ:!

_output_shapes	
:ђ:&"
 
_output_shapes
:
ђђ:!

_output_shapes	
:ђ:&"
 
_output_shapes
:
ђђ:!

_output_shapes	
:ђ:&"
 
_output_shapes
:
ђђ:!

_output_shapes	
:ђ:&"
 
_output_shapes
:
ђђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:! 

_output_shapes	
:ђ:!!

_output_shapes	
:ђ:!"

_output_shapes	
:ђ:!#

_output_shapes	
:ђ:!$

_output_shapes	
:ђ:!%

_output_shapes	
:ђ:!&

_output_shapes	
:ђ:!'

_output_shapes	
:ђ:!(

_output_shapes	
:ђ:!)

_output_shapes	
:ђ:!*

_output_shapes	
:ђ:+

_output_shapes
: 
ў	
█
B__inference_dense_3_layer_call_and_return_conditional_losses_61441

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddќ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
љ■
р"
 __inference__wrapped_model_60252

args_0
args_0_1	
args_0_2
args_0_3	
args_0_4	7
3model_ecc_conv_fgn_0_matmul_readvariableop_resource7
3model_ecc_conv_fgn_1_matmul_readvariableop_resource7
3model_ecc_conv_fgn_2_matmul_readvariableop_resource9
5model_ecc_conv_fgn_out_matmul_readvariableop_resource:
6model_ecc_conv_fgn_out_biasadd_readvariableop_resource3
/model_ecc_conv_matmul_1_readvariableop_resource1
-model_gcn_conv_matmul_readvariableop_resource3
/model_gcn_conv_1_matmul_readvariableop_resource3
/model_gcn_conv_2_matmul_readvariableop_resource3
/model_gcn_conv_3_matmul_readvariableop_resource.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource:
6model_batch_normalization_cast_readvariableop_resource<
8model_batch_normalization_cast_1_readvariableop_resource<
8model_batch_normalization_cast_2_readvariableop_resource<
8model_batch_normalization_cast_3_readvariableop_resource0
,model_dense_1_matmul_readvariableop_resource1
-model_dense_1_biasadd_readvariableop_resource<
8model_batch_normalization_1_cast_readvariableop_resource>
:model_batch_normalization_1_cast_1_readvariableop_resource>
:model_batch_normalization_1_cast_2_readvariableop_resource>
:model_batch_normalization_1_cast_3_readvariableop_resource0
,model_dense_2_matmul_readvariableop_resource1
-model_dense_2_biasadd_readvariableop_resource<
8model_batch_normalization_2_cast_readvariableop_resource>
:model_batch_normalization_2_cast_1_readvariableop_resource>
:model_batch_normalization_2_cast_2_readvariableop_resource>
:model_batch_normalization_2_cast_3_readvariableop_resource0
,model_dense_3_matmul_readvariableop_resource1
-model_dense_3_biasadd_readvariableop_resource<
8model_batch_normalization_3_cast_readvariableop_resource>
:model_batch_normalization_3_cast_1_readvariableop_resource>
:model_batch_normalization_3_cast_2_readvariableop_resource>
:model_batch_normalization_3_cast_3_readvariableop_resource0
,model_dense_4_matmul_readvariableop_resource1
-model_dense_4_biasadd_readvariableop_resource<
8model_batch_normalization_4_cast_readvariableop_resource>
:model_batch_normalization_4_cast_1_readvariableop_resource>
:model_batch_normalization_4_cast_2_readvariableop_resource>
:model_batch_normalization_4_cast_3_readvariableop_resource0
,model_dense_5_matmul_readvariableop_resource1
-model_dense_5_biasadd_readvariableop_resource
identityѕб-model/batch_normalization/Cast/ReadVariableOpб/model/batch_normalization/Cast_1/ReadVariableOpб/model/batch_normalization/Cast_2/ReadVariableOpб/model/batch_normalization/Cast_3/ReadVariableOpб/model/batch_normalization_1/Cast/ReadVariableOpб1model/batch_normalization_1/Cast_1/ReadVariableOpб1model/batch_normalization_1/Cast_2/ReadVariableOpб1model/batch_normalization_1/Cast_3/ReadVariableOpб/model/batch_normalization_2/Cast/ReadVariableOpб1model/batch_normalization_2/Cast_1/ReadVariableOpб1model/batch_normalization_2/Cast_2/ReadVariableOpб1model/batch_normalization_2/Cast_3/ReadVariableOpб/model/batch_normalization_3/Cast/ReadVariableOpб1model/batch_normalization_3/Cast_1/ReadVariableOpб1model/batch_normalization_3/Cast_2/ReadVariableOpб1model/batch_normalization_3/Cast_3/ReadVariableOpб/model/batch_normalization_4/Cast/ReadVariableOpб1model/batch_normalization_4/Cast_1/ReadVariableOpб1model/batch_normalization_4/Cast_2/ReadVariableOpб1model/batch_normalization_4/Cast_3/ReadVariableOpб"model/dense/BiasAdd/ReadVariableOpб!model/dense/MatMul/ReadVariableOpб$model/dense_1/BiasAdd/ReadVariableOpб#model/dense_1/MatMul/ReadVariableOpб$model/dense_2/BiasAdd/ReadVariableOpб#model/dense_2/MatMul/ReadVariableOpб$model/dense_3/BiasAdd/ReadVariableOpб#model/dense_3/MatMul/ReadVariableOpб$model/dense_4/BiasAdd/ReadVariableOpб#model/dense_4/MatMul/ReadVariableOpб$model/dense_5/BiasAdd/ReadVariableOpб#model/dense_5/MatMul/ReadVariableOpб*model/ecc_conv/FGN_0/MatMul/ReadVariableOpб*model/ecc_conv/FGN_1/MatMul/ReadVariableOpб*model/ecc_conv/FGN_2/MatMul/ReadVariableOpб-model/ecc_conv/FGN_out/BiasAdd/ReadVariableOpб,model/ecc_conv/FGN_out/MatMul/ReadVariableOpб&model/ecc_conv/MatMul_1/ReadVariableOpб$model/gcn_conv/MatMul/ReadVariableOpб&model/gcn_conv_1/MatMul/ReadVariableOpб&model/gcn_conv_2/MatMul/ReadVariableOpб&model/gcn_conv_3/MatMul/ReadVariableOpЄ
model/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
model/strided_slice/stackІ
model/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
model/strided_slice/stack_1І
model/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
model/strided_slice/stack_2Е
model/strided_sliceStridedSliceargs_0_1"model/strided_slice/stack:output:0$model/strided_slice/stack_1:output:0$model/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
model/strided_sliceІ
model/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
model/strided_slice_1/stackЈ
model/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
model/strided_slice_1/stack_1Ј
model/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
model/strided_slice_1/stack_2│
model/strided_slice_1StridedSliceargs_0_1$model/strided_slice_1/stack:output:0&model/strided_slice_1/stack_1:output:0&model/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
model/strided_slice_1l
model/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
model/GatherV2/axisК
model/GatherV2GatherV2args_0model/strided_slice_1:output:0model/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:         2
model/GatherV2p
model/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
model/GatherV2_1/axis╦
model/GatherV2_1GatherV2args_0model/strided_slice:output:0model/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:         2
model/GatherV2_1Ѓ
	model/SubSubmodel/GatherV2:output:0model/GatherV2_1:output:0*
T0*'
_output_shapes
:         2
	model/SubІ
model/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
model/strided_slice_2/stackЈ
model/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
model/strided_slice_2/stack_1Ј
model/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
model/strided_slice_2/stack_2ц
model/strided_slice_2StridedSlicemodel/Sub:z:0$model/strided_slice_2/stack:output:0&model/strided_slice_2/stack_1:output:0&model/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
model/strided_slice_2x
model/SquareSquaremodel/strided_slice_2:output:0*
T0*'
_output_shapes
:         2
model/Square|
model/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
model/Sum/reduction_indicesЃ
	model/SumSummodel/Square:y:0$model/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
	model/Sumb

model/SqrtSqrtmodel/Sum:output:0*
T0*#
_output_shapes
:         2

model/SqrtІ
model/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
model/strided_slice_3/stackЈ
model/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
model/strided_slice_3/stack_1Ј
model/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
model/strided_slice_3/stack_2ц
model/strided_slice_3StridedSlicemodel/Sub:z:0$model/strided_slice_3/stack:output:0&model/strided_slice_3/stack_1:output:0&model/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
model/strided_slice_3w
model/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         2
model/ExpandDims/dimЊ
model/ExpandDims
ExpandDimsmodel/Sqrt:y:0model/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         2
model/ExpandDimsЮ
model/div_no_nanDivNoNanmodel/strided_slice_3:output:0model/ExpandDims:output:0*
T0*'
_output_shapes
:         2
model/div_no_nanІ
model/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
model/strided_slice_4/stackЈ
model/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
model/strided_slice_4/stack_1Ј
model/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
model/strided_slice_4/stack_2ц
model/strided_slice_4StridedSlicemodel/Sub:z:0$model/strided_slice_4/stack:output:0&model/strided_slice_4/stack_1:output:0&model/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
model/strided_slice_4{
model/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
         2
model/ExpandDims_1/dimЎ
model/ExpandDims_1
ExpandDimsmodel/Sqrt:y:0model/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:         2
model/ExpandDims_1h
model/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concat/axisм
model/concatConcatV2model/strided_slice_4:output:0model/ExpandDims_1:output:0model/div_no_nan:z:0model/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
model/concatb
model/ecc_conv/ShapeShapeargs_0*
T0*
_output_shapes
:2
model/ecc_conv/ShapeЏ
"model/ecc_conv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        2$
"model/ecc_conv/strided_slice/stackЪ
$model/ecc_conv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         2&
$model/ecc_conv/strided_slice/stack_1ќ
$model/ecc_conv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$model/ecc_conv/strided_slice/stack_2╝
model/ecc_conv/strided_sliceStridedSlicemodel/ecc_conv/Shape:output:0+model/ecc_conv/strided_slice/stack:output:0-model/ecc_conv/strided_slice/stack_1:output:0-model/ecc_conv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/ecc_conv/strided_slice╠
*model/ecc_conv/FGN_0/MatMul/ReadVariableOpReadVariableOp3model_ecc_conv_fgn_0_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02,
*model/ecc_conv/FGN_0/MatMul/ReadVariableOp┴
model/ecc_conv/FGN_0/MatMulMatMulmodel/concat:output:02model/ecc_conv/FGN_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
model/ecc_conv/FGN_0/MatMulЌ
model/ecc_conv/FGN_0/ReluRelu%model/ecc_conv/FGN_0/MatMul:product:0*
T0*'
_output_shapes
:         @2
model/ecc_conv/FGN_0/Relu╠
*model/ecc_conv/FGN_1/MatMul/ReadVariableOpReadVariableOp3model_ecc_conv_fgn_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02,
*model/ecc_conv/FGN_1/MatMul/ReadVariableOpМ
model/ecc_conv/FGN_1/MatMulMatMul'model/ecc_conv/FGN_0/Relu:activations:02model/ecc_conv/FGN_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
model/ecc_conv/FGN_1/MatMulЌ
model/ecc_conv/FGN_1/ReluRelu%model/ecc_conv/FGN_1/MatMul:product:0*
T0*'
_output_shapes
:         @2
model/ecc_conv/FGN_1/Relu╠
*model/ecc_conv/FGN_2/MatMul/ReadVariableOpReadVariableOp3model_ecc_conv_fgn_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02,
*model/ecc_conv/FGN_2/MatMul/ReadVariableOpМ
model/ecc_conv/FGN_2/MatMulMatMul'model/ecc_conv/FGN_1/Relu:activations:02model/ecc_conv/FGN_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
model/ecc_conv/FGN_2/MatMulЌ
model/ecc_conv/FGN_2/ReluRelu%model/ecc_conv/FGN_2/MatMul:product:0*
T0*'
_output_shapes
:         @2
model/ecc_conv/FGN_2/ReluМ
,model/ecc_conv/FGN_out/MatMul/ReadVariableOpReadVariableOp5model_ecc_conv_fgn_out_matmul_readvariableop_resource*
_output_shapes
:	@└*
dtype02.
,model/ecc_conv/FGN_out/MatMul/ReadVariableOp┌
model/ecc_conv/FGN_out/MatMulMatMul'model/ecc_conv/FGN_2/Relu:activations:04model/ecc_conv/FGN_out/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └2
model/ecc_conv/FGN_out/MatMulм
-model/ecc_conv/FGN_out/BiasAdd/ReadVariableOpReadVariableOp6model_ecc_conv_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:└*
dtype02/
-model/ecc_conv/FGN_out/BiasAdd/ReadVariableOpя
model/ecc_conv/FGN_out/BiasAddBiasAdd'model/ecc_conv/FGN_out/MatMul:product:05model/ecc_conv/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └2 
model/ecc_conv/FGN_out/BiasAddЉ
model/ecc_conv/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       @   2
model/ecc_conv/Reshape/shape┴
model/ecc_conv/ReshapeReshape'model/ecc_conv/FGN_out/BiasAdd:output:0%model/ecc_conv/Reshape/shape:output:0*
T0*+
_output_shapes
:         @2
model/ecc_conv/ReshapeЮ
$model/ecc_conv/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ■   2&
$model/ecc_conv/strided_slice_1/stackА
&model/ecc_conv/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&model/ecc_conv/strided_slice_1/stack_1А
&model/ecc_conv/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&model/ecc_conv/strided_slice_1/stack_2Я
model/ecc_conv/strided_slice_1StridedSliceargs_0_1-model/ecc_conv/strided_slice_1/stack:output:0/model/ecc_conv/strided_slice_1/stack_1:output:0/model/ecc_conv/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2 
model/ecc_conv/strided_slice_1Ю
$model/ecc_conv/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$model/ecc_conv/strided_slice_2/stackА
&model/ecc_conv/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&model/ecc_conv/strided_slice_2/stack_1А
&model/ecc_conv/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&model/ecc_conv/strided_slice_2/stack_2Я
model/ecc_conv/strided_slice_2StridedSliceargs_0_1-model/ecc_conv/strided_slice_2/stack:output:0/model/ecc_conv/strided_slice_2/stack_1:output:0/model/ecc_conv/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2 
model/ecc_conv/strided_slice_2~
model/ecc_conv/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
model/ecc_conv/GatherV2/axisв
model/ecc_conv/GatherV2GatherV2args_0'model/ecc_conv/strided_slice_2:output:0%model/ecc_conv/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:         2
model/ecc_conv/GatherV2А
$model/ecc_conv/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2&
$model/ecc_conv/strided_slice_3/stackЦ
&model/ecc_conv/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2(
&model/ecc_conv/strided_slice_3/stack_1Ц
&model/ecc_conv/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2(
&model/ecc_conv/strided_slice_3/stack_2§
model/ecc_conv/strided_slice_3StridedSlice model/ecc_conv/GatherV2:output:0-model/ecc_conv/strided_slice_3/stack:output:0/model/ecc_conv/strided_slice_3/stack_1:output:0/model/ecc_conv/strided_slice_3/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2 
model/ecc_conv/strided_slice_3┐
model/ecc_conv/MatMulBatchMatMulV2'model/ecc_conv/strided_slice_3:output:0model/ecc_conv/Reshape:output:0*
T0*+
_output_shapes
:         @2
model/ecc_conv/MatMulА
$model/ecc_conv/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2&
$model/ecc_conv/strided_slice_4/stackЦ
&model/ecc_conv/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2(
&model/ecc_conv/strided_slice_4/stack_1Ц
&model/ecc_conv/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2(
&model/ecc_conv/strided_slice_4/stack_2Щ
model/ecc_conv/strided_slice_4StridedSlicemodel/ecc_conv/MatMul:output:0-model/ecc_conv/strided_slice_4/stack:output:0/model/ecc_conv/strided_slice_4/stack_1:output:0/model/ecc_conv/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*

begin_mask*
end_mask*
shrink_axis_mask2 
model/ecc_conv/strided_slice_4Ќ
!model/ecc_conv/UnsortedSegmentSumUnsortedSegmentSum'model/ecc_conv/strided_slice_4:output:0'model/ecc_conv/strided_slice_1:output:0%model/ecc_conv/strided_slice:output:0*
T0*
Tindices0	*'
_output_shapes
:         @2#
!model/ecc_conv/UnsortedSegmentSum└
&model/ecc_conv/MatMul_1/ReadVariableOpReadVariableOp/model_ecc_conv_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype02(
&model/ecc_conv/MatMul_1/ReadVariableOpд
model/ecc_conv/MatMul_1MatMulargs_0.model/ecc_conv/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
model/ecc_conv/MatMul_1▓
model/ecc_conv/addAddV2*model/ecc_conv/UnsortedSegmentSum:output:0!model/ecc_conv/MatMul_1:product:0*
T0*'
_output_shapes
:         @2
model/ecc_conv/add|
model/ecc_conv/ReluRelumodel/ecc_conv/add:z:0*
T0*'
_output_shapes
:         @2
model/ecc_conv/Relu║
$model/gcn_conv/MatMul/ReadVariableOpReadVariableOp-model_gcn_conv_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02&
$model/gcn_conv/MatMul/ReadVariableOp╗
model/gcn_conv/MatMulMatMul!model/ecc_conv/Relu:activations:0,model/gcn_conv/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
model/gcn_conv/MatMulї
>model/gcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulargs_0_1args_0_2args_0_3model/gcn_conv/MatMul:product:0*
T0*'
_output_shapes
:         @2@
>model/gcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMul«
model/gcn_conv/ReluReluHmodel/gcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*'
_output_shapes
:         @2
model/gcn_conv/Relu┴
&model/gcn_conv_1/MatMul/ReadVariableOpReadVariableOp/model_gcn_conv_1_matmul_readvariableop_resource*
_output_shapes
:	@ђ*
dtype02(
&model/gcn_conv_1/MatMul/ReadVariableOp┬
model/gcn_conv_1/MatMulMatMul!model/gcn_conv/Relu:activations:0.model/gcn_conv_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
model/gcn_conv_1/MatMulЊ
@model/gcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulargs_0_1args_0_2args_0_3!model/gcn_conv_1/MatMul:product:0*
T0*(
_output_shapes
:         ђ2B
@model/gcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMulх
model/gcn_conv_1/ReluReluJmodel/gcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:         ђ2
model/gcn_conv_1/Relu┬
&model/gcn_conv_2/MatMul/ReadVariableOpReadVariableOp/model_gcn_conv_2_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02(
&model/gcn_conv_2/MatMul/ReadVariableOp─
model/gcn_conv_2/MatMulMatMul#model/gcn_conv_1/Relu:activations:0.model/gcn_conv_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
model/gcn_conv_2/MatMulЊ
@model/gcn_conv_2/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulargs_0_1args_0_2args_0_3!model/gcn_conv_2/MatMul:product:0*
T0*(
_output_shapes
:         ђ2B
@model/gcn_conv_2/SparseTensorDenseMatMul/SparseTensorDenseMatMulх
model/gcn_conv_2/ReluReluJmodel/gcn_conv_2/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:         ђ2
model/gcn_conv_2/Relu┬
&model/gcn_conv_3/MatMul/ReadVariableOpReadVariableOp/model_gcn_conv_3_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02(
&model/gcn_conv_3/MatMul/ReadVariableOp─
model/gcn_conv_3/MatMulMatMul#model/gcn_conv_2/Relu:activations:0.model/gcn_conv_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
model/gcn_conv_3/MatMulЊ
@model/gcn_conv_3/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulargs_0_1args_0_2args_0_3!model/gcn_conv_3/MatMul:product:0*
T0*(
_output_shapes
:         ђ2B
@model/gcn_conv_3/SparseTensorDenseMatMul/SparseTensorDenseMatMulх
model/gcn_conv_3/ReluReluJmodel/gcn_conv_3/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:         ђ2
model/gcn_conv_3/Relu─
 model/global_max_pool/SegmentMax
SegmentMax#model/gcn_conv_3/Relu:activations:0args_0_4*
T0*
Tindices0	*(
_output_shapes
:         ђ2"
 model/global_max_pool/SegmentMaxК
!model/global_avg_pool/SegmentMeanSegmentMean#model/gcn_conv_3/Relu:activations:0args_0_4*
T0*
Tindices0	*(
_output_shapes
:         ђ2#
!model/global_avg_pool/SegmentMean─
 model/global_sum_pool/SegmentSum
SegmentSum#model/gcn_conv_3/Relu:activations:0args_0_4*
T0*
Tindices0	*(
_output_shapes
:         ђ2"
 model/global_sum_pool/SegmentSuml
model/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concat_1/axisѕ
model/concat_1ConcatV2)model/global_max_pool/SegmentMax:output:0*model/global_avg_pool/SegmentMean:output:0)model/global_sum_pool/SegmentSum:output:0model/concat_1/axis:output:0*
N*
T0*(
_output_shapes
:         ђ2
model/concat_1│
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02#
!model/dense/MatMul/ReadVariableOpЕ
model/dense/MatMulMatMulmodel/concat_1:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
model/dense/MatMul▒
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02$
"model/dense/BiasAdd/ReadVariableOp▓
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
model/dense/BiasAddЪ
model/leaky_re_lu/LeakyRelu	LeakyRelumodel/dense/BiasAdd:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
model/leaky_re_lu/LeakyReluм
-model/batch_normalization/Cast/ReadVariableOpReadVariableOp6model_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-model/batch_normalization/Cast/ReadVariableOpп
/model/batch_normalization/Cast_1/ReadVariableOpReadVariableOp8model_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype021
/model/batch_normalization/Cast_1/ReadVariableOpп
/model/batch_normalization/Cast_2/ReadVariableOpReadVariableOp8model_batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype021
/model/batch_normalization/Cast_2/ReadVariableOpп
/model/batch_normalization/Cast_3/ReadVariableOpReadVariableOp8model_batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype021
/model/batch_normalization/Cast_3/ReadVariableOpЏ
)model/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2+
)model/batch_normalization/batchnorm/add/yЬ
'model/batch_normalization/batchnorm/addAddV27model/batch_normalization/Cast_1/ReadVariableOp:value:02model/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2)
'model/batch_normalization/batchnorm/add▓
)model/batch_normalization/batchnorm/RsqrtRsqrt+model/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2+
)model/batch_normalization/batchnorm/Rsqrtу
'model/batch_normalization/batchnorm/mulMul-model/batch_normalization/batchnorm/Rsqrt:y:07model/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2)
'model/batch_normalization/batchnorm/mulУ
)model/batch_normalization/batchnorm/mul_1Mul)model/leaky_re_lu/LeakyRelu:activations:0+model/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2+
)model/batch_normalization/batchnorm/mul_1у
)model/batch_normalization/batchnorm/mul_2Mul5model/batch_normalization/Cast/ReadVariableOp:value:0+model/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2+
)model/batch_normalization/batchnorm/mul_2у
'model/batch_normalization/batchnorm/subSub7model/batch_normalization/Cast_2/ReadVariableOp:value:0-model/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2)
'model/batch_normalization/batchnorm/subЬ
)model/batch_normalization/batchnorm/add_1AddV2-model/batch_normalization/batchnorm/mul_1:z:0+model/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2+
)model/batch_normalization/batchnorm/add_1╣
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02%
#model/dense_1/MatMul/ReadVariableOp┼
model/dense_1/MatMulMatMul-model/batch_normalization/batchnorm/add_1:z:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
model/dense_1/MatMulи
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp║
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
model/dense_1/BiasAddЦ
model/leaky_re_lu/LeakyRelu_1	LeakyRelumodel/dense_1/BiasAdd:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
model/leaky_re_lu/LeakyRelu_1п
/model/batch_normalization_1/Cast/ReadVariableOpReadVariableOp8model_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:ђ*
dtype021
/model/batch_normalization_1/Cast/ReadVariableOpя
1model/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp:model_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype023
1model/batch_normalization_1/Cast_1/ReadVariableOpя
1model/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp:model_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype023
1model/batch_normalization_1/Cast_2/ReadVariableOpя
1model/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp:model_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype023
1model/batch_normalization_1/Cast_3/ReadVariableOpЪ
+model/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2-
+model/batch_normalization_1/batchnorm/add/yШ
)model/batch_normalization_1/batchnorm/addAddV29model/batch_normalization_1/Cast_1/ReadVariableOp:value:04model/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2+
)model/batch_normalization_1/batchnorm/addИ
+model/batch_normalization_1/batchnorm/RsqrtRsqrt-model/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2-
+model/batch_normalization_1/batchnorm/Rsqrt№
)model/batch_normalization_1/batchnorm/mulMul/model/batch_normalization_1/batchnorm/Rsqrt:y:09model/batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2+
)model/batch_normalization_1/batchnorm/mul­
+model/batch_normalization_1/batchnorm/mul_1Mul+model/leaky_re_lu/LeakyRelu_1:activations:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2-
+model/batch_normalization_1/batchnorm/mul_1№
+model/batch_normalization_1/batchnorm/mul_2Mul7model/batch_normalization_1/Cast/ReadVariableOp:value:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2-
+model/batch_normalization_1/batchnorm/mul_2№
)model/batch_normalization_1/batchnorm/subSub9model/batch_normalization_1/Cast_2/ReadVariableOp:value:0/model/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2+
)model/batch_normalization_1/batchnorm/subШ
+model/batch_normalization_1/batchnorm/add_1AddV2/model/batch_normalization_1/batchnorm/mul_1:z:0-model/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2-
+model/batch_normalization_1/batchnorm/add_1╣
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02%
#model/dense_2/MatMul/ReadVariableOpК
model/dense_2/MatMulMatMul/model/batch_normalization_1/batchnorm/add_1:z:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
model/dense_2/MatMulи
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$model/dense_2/BiasAdd/ReadVariableOp║
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
model/dense_2/BiasAddЦ
model/leaky_re_lu/LeakyRelu_2	LeakyRelumodel/dense_2/BiasAdd:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
model/leaky_re_lu/LeakyRelu_2п
/model/batch_normalization_2/Cast/ReadVariableOpReadVariableOp8model_batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:ђ*
dtype021
/model/batch_normalization_2/Cast/ReadVariableOpя
1model/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp:model_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype023
1model/batch_normalization_2/Cast_1/ReadVariableOpя
1model/batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp:model_batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype023
1model/batch_normalization_2/Cast_2/ReadVariableOpя
1model/batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp:model_batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype023
1model/batch_normalization_2/Cast_3/ReadVariableOpЪ
+model/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2-
+model/batch_normalization_2/batchnorm/add/yШ
)model/batch_normalization_2/batchnorm/addAddV29model/batch_normalization_2/Cast_1/ReadVariableOp:value:04model/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2+
)model/batch_normalization_2/batchnorm/addИ
+model/batch_normalization_2/batchnorm/RsqrtRsqrt-model/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2-
+model/batch_normalization_2/batchnorm/Rsqrt№
)model/batch_normalization_2/batchnorm/mulMul/model/batch_normalization_2/batchnorm/Rsqrt:y:09model/batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2+
)model/batch_normalization_2/batchnorm/mul­
+model/batch_normalization_2/batchnorm/mul_1Mul+model/leaky_re_lu/LeakyRelu_2:activations:0-model/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2-
+model/batch_normalization_2/batchnorm/mul_1№
+model/batch_normalization_2/batchnorm/mul_2Mul7model/batch_normalization_2/Cast/ReadVariableOp:value:0-model/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2-
+model/batch_normalization_2/batchnorm/mul_2№
)model/batch_normalization_2/batchnorm/subSub9model/batch_normalization_2/Cast_2/ReadVariableOp:value:0/model/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2+
)model/batch_normalization_2/batchnorm/subШ
+model/batch_normalization_2/batchnorm/add_1AddV2/model/batch_normalization_2/batchnorm/mul_1:z:0-model/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2-
+model/batch_normalization_2/batchnorm/add_1╣
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02%
#model/dense_3/MatMul/ReadVariableOpК
model/dense_3/MatMulMatMul/model/batch_normalization_2/batchnorm/add_1:z:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
model/dense_3/MatMulи
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$model/dense_3/BiasAdd/ReadVariableOp║
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
model/dense_3/BiasAddЦ
model/leaky_re_lu/LeakyRelu_3	LeakyRelumodel/dense_3/BiasAdd:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
model/leaky_re_lu/LeakyRelu_3п
/model/batch_normalization_3/Cast/ReadVariableOpReadVariableOp8model_batch_normalization_3_cast_readvariableop_resource*
_output_shapes	
:ђ*
dtype021
/model/batch_normalization_3/Cast/ReadVariableOpя
1model/batch_normalization_3/Cast_1/ReadVariableOpReadVariableOp:model_batch_normalization_3_cast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype023
1model/batch_normalization_3/Cast_1/ReadVariableOpя
1model/batch_normalization_3/Cast_2/ReadVariableOpReadVariableOp:model_batch_normalization_3_cast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype023
1model/batch_normalization_3/Cast_2/ReadVariableOpя
1model/batch_normalization_3/Cast_3/ReadVariableOpReadVariableOp:model_batch_normalization_3_cast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype023
1model/batch_normalization_3/Cast_3/ReadVariableOpЪ
+model/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2-
+model/batch_normalization_3/batchnorm/add/yШ
)model/batch_normalization_3/batchnorm/addAddV29model/batch_normalization_3/Cast_1/ReadVariableOp:value:04model/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2+
)model/batch_normalization_3/batchnorm/addИ
+model/batch_normalization_3/batchnorm/RsqrtRsqrt-model/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2-
+model/batch_normalization_3/batchnorm/Rsqrt№
)model/batch_normalization_3/batchnorm/mulMul/model/batch_normalization_3/batchnorm/Rsqrt:y:09model/batch_normalization_3/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2+
)model/batch_normalization_3/batchnorm/mul­
+model/batch_normalization_3/batchnorm/mul_1Mul+model/leaky_re_lu/LeakyRelu_3:activations:0-model/batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2-
+model/batch_normalization_3/batchnorm/mul_1№
+model/batch_normalization_3/batchnorm/mul_2Mul7model/batch_normalization_3/Cast/ReadVariableOp:value:0-model/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2-
+model/batch_normalization_3/batchnorm/mul_2№
)model/batch_normalization_3/batchnorm/subSub9model/batch_normalization_3/Cast_2/ReadVariableOp:value:0/model/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2+
)model/batch_normalization_3/batchnorm/subШ
+model/batch_normalization_3/batchnorm/add_1AddV2/model/batch_normalization_3/batchnorm/mul_1:z:0-model/batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2-
+model/batch_normalization_3/batchnorm/add_1╣
#model/dense_4/MatMul/ReadVariableOpReadVariableOp,model_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02%
#model/dense_4/MatMul/ReadVariableOpК
model/dense_4/MatMulMatMul/model/batch_normalization_3/batchnorm/add_1:z:0+model/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
model/dense_4/MatMulи
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$model/dense_4/BiasAdd/ReadVariableOp║
model/dense_4/BiasAddBiasAddmodel/dense_4/MatMul:product:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
model/dense_4/BiasAddЦ
model/leaky_re_lu/LeakyRelu_4	LeakyRelumodel/dense_4/BiasAdd:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
model/leaky_re_lu/LeakyRelu_4п
/model/batch_normalization_4/Cast/ReadVariableOpReadVariableOp8model_batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:ђ*
dtype021
/model/batch_normalization_4/Cast/ReadVariableOpя
1model/batch_normalization_4/Cast_1/ReadVariableOpReadVariableOp:model_batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype023
1model/batch_normalization_4/Cast_1/ReadVariableOpя
1model/batch_normalization_4/Cast_2/ReadVariableOpReadVariableOp:model_batch_normalization_4_cast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype023
1model/batch_normalization_4/Cast_2/ReadVariableOpя
1model/batch_normalization_4/Cast_3/ReadVariableOpReadVariableOp:model_batch_normalization_4_cast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype023
1model/batch_normalization_4/Cast_3/ReadVariableOpЪ
+model/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2-
+model/batch_normalization_4/batchnorm/add/yШ
)model/batch_normalization_4/batchnorm/addAddV29model/batch_normalization_4/Cast_1/ReadVariableOp:value:04model/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2+
)model/batch_normalization_4/batchnorm/addИ
+model/batch_normalization_4/batchnorm/RsqrtRsqrt-model/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:ђ2-
+model/batch_normalization_4/batchnorm/Rsqrt№
)model/batch_normalization_4/batchnorm/mulMul/model/batch_normalization_4/batchnorm/Rsqrt:y:09model/batch_normalization_4/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2+
)model/batch_normalization_4/batchnorm/mul­
+model/batch_normalization_4/batchnorm/mul_1Mul+model/leaky_re_lu/LeakyRelu_4:activations:0-model/batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2-
+model/batch_normalization_4/batchnorm/mul_1№
+model/batch_normalization_4/batchnorm/mul_2Mul7model/batch_normalization_4/Cast/ReadVariableOp:value:0-model/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2-
+model/batch_normalization_4/batchnorm/mul_2№
)model/batch_normalization_4/batchnorm/subSub9model/batch_normalization_4/Cast_2/ReadVariableOp:value:0/model/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2+
)model/batch_normalization_4/batchnorm/subШ
+model/batch_normalization_4/batchnorm/add_1AddV2/model/batch_normalization_4/batchnorm/mul_1:z:0-model/batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2-
+model/batch_normalization_4/batchnorm/add_1И
#model/dense_5/MatMul/ReadVariableOpReadVariableOp,model_dense_5_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02%
#model/dense_5/MatMul/ReadVariableOpк
model/dense_5/MatMulMatMul/model/batch_normalization_4/batchnorm/add_1:z:0+model/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/dense_5/MatMulХ
$model/dense_5/BiasAdd/ReadVariableOpReadVariableOp-model_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_5/BiasAdd/ReadVariableOp╣
model/dense_5/BiasAddBiasAddmodel/dense_5/MatMul:product:0,model/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/dense_5/BiasAddв
IdentityIdentitymodel/dense_5/BiasAdd:output:0.^model/batch_normalization/Cast/ReadVariableOp0^model/batch_normalization/Cast_1/ReadVariableOp0^model/batch_normalization/Cast_2/ReadVariableOp0^model/batch_normalization/Cast_3/ReadVariableOp0^model/batch_normalization_1/Cast/ReadVariableOp2^model/batch_normalization_1/Cast_1/ReadVariableOp2^model/batch_normalization_1/Cast_2/ReadVariableOp2^model/batch_normalization_1/Cast_3/ReadVariableOp0^model/batch_normalization_2/Cast/ReadVariableOp2^model/batch_normalization_2/Cast_1/ReadVariableOp2^model/batch_normalization_2/Cast_2/ReadVariableOp2^model/batch_normalization_2/Cast_3/ReadVariableOp0^model/batch_normalization_3/Cast/ReadVariableOp2^model/batch_normalization_3/Cast_1/ReadVariableOp2^model/batch_normalization_3/Cast_2/ReadVariableOp2^model/batch_normalization_3/Cast_3/ReadVariableOp0^model/batch_normalization_4/Cast/ReadVariableOp2^model/batch_normalization_4/Cast_1/ReadVariableOp2^model/batch_normalization_4/Cast_2/ReadVariableOp2^model/batch_normalization_4/Cast_3/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp$^model/dense_4/MatMul/ReadVariableOp%^model/dense_5/BiasAdd/ReadVariableOp$^model/dense_5/MatMul/ReadVariableOp+^model/ecc_conv/FGN_0/MatMul/ReadVariableOp+^model/ecc_conv/FGN_1/MatMul/ReadVariableOp+^model/ecc_conv/FGN_2/MatMul/ReadVariableOp.^model/ecc_conv/FGN_out/BiasAdd/ReadVariableOp-^model/ecc_conv/FGN_out/MatMul/ReadVariableOp'^model/ecc_conv/MatMul_1/ReadVariableOp%^model/gcn_conv/MatMul/ReadVariableOp'^model/gcn_conv_1/MatMul/ReadVariableOp'^model/gcn_conv_2/MatMul/ReadVariableOp'^model/gcn_conv_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Є
_input_shapesш
Ы:         :         :         ::         ::::::::::::::::::::::::::::::::::::::::::2^
-model/batch_normalization/Cast/ReadVariableOp-model/batch_normalization/Cast/ReadVariableOp2b
/model/batch_normalization/Cast_1/ReadVariableOp/model/batch_normalization/Cast_1/ReadVariableOp2b
/model/batch_normalization/Cast_2/ReadVariableOp/model/batch_normalization/Cast_2/ReadVariableOp2b
/model/batch_normalization/Cast_3/ReadVariableOp/model/batch_normalization/Cast_3/ReadVariableOp2b
/model/batch_normalization_1/Cast/ReadVariableOp/model/batch_normalization_1/Cast/ReadVariableOp2f
1model/batch_normalization_1/Cast_1/ReadVariableOp1model/batch_normalization_1/Cast_1/ReadVariableOp2f
1model/batch_normalization_1/Cast_2/ReadVariableOp1model/batch_normalization_1/Cast_2/ReadVariableOp2f
1model/batch_normalization_1/Cast_3/ReadVariableOp1model/batch_normalization_1/Cast_3/ReadVariableOp2b
/model/batch_normalization_2/Cast/ReadVariableOp/model/batch_normalization_2/Cast/ReadVariableOp2f
1model/batch_normalization_2/Cast_1/ReadVariableOp1model/batch_normalization_2/Cast_1/ReadVariableOp2f
1model/batch_normalization_2/Cast_2/ReadVariableOp1model/batch_normalization_2/Cast_2/ReadVariableOp2f
1model/batch_normalization_2/Cast_3/ReadVariableOp1model/batch_normalization_2/Cast_3/ReadVariableOp2b
/model/batch_normalization_3/Cast/ReadVariableOp/model/batch_normalization_3/Cast/ReadVariableOp2f
1model/batch_normalization_3/Cast_1/ReadVariableOp1model/batch_normalization_3/Cast_1/ReadVariableOp2f
1model/batch_normalization_3/Cast_2/ReadVariableOp1model/batch_normalization_3/Cast_2/ReadVariableOp2f
1model/batch_normalization_3/Cast_3/ReadVariableOp1model/batch_normalization_3/Cast_3/ReadVariableOp2b
/model/batch_normalization_4/Cast/ReadVariableOp/model/batch_normalization_4/Cast/ReadVariableOp2f
1model/batch_normalization_4/Cast_1/ReadVariableOp1model/batch_normalization_4/Cast_1/ReadVariableOp2f
1model/batch_normalization_4/Cast_2/ReadVariableOp1model/batch_normalization_4/Cast_2/ReadVariableOp2f
1model/batch_normalization_4/Cast_3/ReadVariableOp1model/batch_normalization_4/Cast_3/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2J
#model/dense_3/MatMul/ReadVariableOp#model/dense_3/MatMul/ReadVariableOp2L
$model/dense_4/BiasAdd/ReadVariableOp$model/dense_4/BiasAdd/ReadVariableOp2J
#model/dense_4/MatMul/ReadVariableOp#model/dense_4/MatMul/ReadVariableOp2L
$model/dense_5/BiasAdd/ReadVariableOp$model/dense_5/BiasAdd/ReadVariableOp2J
#model/dense_5/MatMul/ReadVariableOp#model/dense_5/MatMul/ReadVariableOp2X
*model/ecc_conv/FGN_0/MatMul/ReadVariableOp*model/ecc_conv/FGN_0/MatMul/ReadVariableOp2X
*model/ecc_conv/FGN_1/MatMul/ReadVariableOp*model/ecc_conv/FGN_1/MatMul/ReadVariableOp2X
*model/ecc_conv/FGN_2/MatMul/ReadVariableOp*model/ecc_conv/FGN_2/MatMul/ReadVariableOp2^
-model/ecc_conv/FGN_out/BiasAdd/ReadVariableOp-model/ecc_conv/FGN_out/BiasAdd/ReadVariableOp2\
,model/ecc_conv/FGN_out/MatMul/ReadVariableOp,model/ecc_conv/FGN_out/MatMul/ReadVariableOp2P
&model/ecc_conv/MatMul_1/ReadVariableOp&model/ecc_conv/MatMul_1/ReadVariableOp2L
$model/gcn_conv/MatMul/ReadVariableOp$model/gcn_conv/MatMul/ReadVariableOp2P
&model/gcn_conv_1/MatMul/ReadVariableOp&model/gcn_conv_1/MatMul/ReadVariableOp2P
&model/gcn_conv_2/MatMul/ReadVariableOp&model/gcn_conv_2/MatMul/ReadVariableOp2P
&model/gcn_conv_3/MatMul/ReadVariableOp&model/gcn_conv_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameargs_0:OK
'
_output_shapes
:         
 
_user_specified_nameargs_0:KG
#
_output_shapes
:         
 
_user_specified_nameargs_0:B>

_output_shapes
:
 
_user_specified_nameargs_0:KG
#
_output_shapes
:         
 
_user_specified_nameargs_0
ї
╚
C__inference_gcn_conv_layer_call_and_return_conditional_losses_61102

inputs
inputs_1	
inputs_2
inputs_3	"
matmul_readvariableop_resource
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
MatMul▀
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1inputs_2inputs_3MatMul:product:0*
T0*'
_output_shapes
:         @21
/SparseTensorDenseMatMul/SparseTensorDenseMatMulЂ
ReluRelu9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*'
_output_shapes
:         @2
Relu~
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:         @:         :         ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
ќ
┘
N__inference_batch_normalization_layer_call_and_return_conditional_losses_60381

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identityѕбCast/ReadVariableOpбCast_1/ReadVariableOpбCast_2/ReadVariableOpбCast_3/ReadVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_1/ReadVariableOpі
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_2/ReadVariableOpі
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yє
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/add_1к
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ћ
╩
E__inference_gcn_conv_1_layer_call_and_return_conditional_losses_63185
inputs_0

inputs	
inputs_1
inputs_2	"
matmul_readvariableop_resource
identityѕбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@ђ*
dtype02
MatMul/ReadVariableOpv
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulя
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2MatMul:product:0*
T0*(
_output_shapes
:         ђ21
/SparseTensorDenseMatMul/SparseTensorDenseMatMulѓ
ReluRelu9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:         ђ2
Relu
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:         @:         :         ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
'
_output_shapes
:         @
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
б/
Г
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_63589

inputs
assignmovingavg_63564
assignmovingavg_1_63570 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб AssignMovingAvg_1/ReadVariableOpбCast/ReadVariableOpбCast_1/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesљ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ђ2
moments/StopGradientЦ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         ђ2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeeze_1╦
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/63564*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayЊ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_63564*
_output_shapes	
:ђ*
dtype02 
AssignMovingAvg/ReadVariableOpы
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/63564*
_output_shapes	
:ђ2
AssignMovingAvg/subУ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/63564*
_output_shapes	
:ђ2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_63564AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/63564*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЛ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/63570*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayЎ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_63570*
_output_shapes	
:ђ*
dtype02"
 AssignMovingAvg_1/ReadVariableOpч
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/63570*
_output_shapes	
:ђ2
AssignMovingAvg_1/subЫ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/63570*
_output_shapes	
:ђ2
AssignMovingAvg_1/mul╣
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_63570AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/63570*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/add_1е
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
б/
Г
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_60908

inputs
assignmovingavg_60883
assignmovingavg_1_60889 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб AssignMovingAvg_1/ReadVariableOpбCast/ReadVariableOpбCast_1/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesљ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ђ2
moments/StopGradientЦ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         ђ2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeeze_1╦
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/60883*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayЊ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_60883*
_output_shapes	
:ђ*
dtype02 
AssignMovingAvg/ReadVariableOpы
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/60883*
_output_shapes	
:ђ2
AssignMovingAvg/subУ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/60883*
_output_shapes	
:ђ2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_60883AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/60883*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЛ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/60889*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayЎ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_60889*
_output_shapes	
:ђ*
dtype02"
 AssignMovingAvg_1/ReadVariableOpч
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/60889*
_output_shapes	
:ђ2
AssignMovingAvg_1/subЫ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/60889*
_output_shapes	
:ђ2
AssignMovingAvg_1/mul╣
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_60889AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/60889*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/add_1е
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ў	
█
B__inference_dense_2_layer_call_and_return_conditional_losses_61379

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddќ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Й
е
5__inference_batch_normalization_2_layer_call_fn_63635

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_606612
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ў	
█
B__inference_dense_2_layer_call_and_return_conditional_losses_63342

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddќ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
█
v
J__inference_global_max_pool_layer_call_and_return_conditional_losses_63245
inputs_0
inputs_1	
identity}

SegmentMax
SegmentMaxinputs_0inputs_1*
T0*
Tindices0	*(
_output_shapes
:         ђ2

SegmentMaxh
IdentityIdentitySegmentMax:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ђ:         :R N
(
_output_shapes
:         ђ
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:         
"
_user_specified_name
inputs/1
╝
е
5__inference_batch_normalization_3_layer_call_fn_63704

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_607682
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ќ
╩
E__inference_gcn_conv_3_layer_call_and_return_conditional_losses_61183

inputs
inputs_1	
inputs_2
inputs_3	"
matmul_readvariableop_resource
identityѕбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЯ
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1inputs_2inputs_3MatMul:product:0*
T0*(
_output_shapes
:         ђ21
/SparseTensorDenseMatMul/SparseTensorDenseMatMulѓ
ReluRelu9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:         ђ2
Relu
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*S
_input_shapesB
@:         ђ:         :         ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
ј
╚
C__inference_gcn_conv_layer_call_and_return_conditional_losses_63163
inputs_0

inputs	
inputs_1
inputs_2	"
matmul_readvariableop_resource
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
MatMulП
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2MatMul:product:0*
T0*'
_output_shapes
:         @21
/SparseTensorDenseMatMul/SparseTensorDenseMatMulЂ
ReluRelu9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*'
_output_shapes
:         @2
Relu~
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:         @:         :         ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
'
_output_shapes
:         @
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
ў
█
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_63691

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identityѕбCast/ReadVariableOpбCast_1/ReadVariableOpбCast_2/ReadVariableOpбCast_3/ReadVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_1/ReadVariableOpі
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_2/ReadVariableOpі
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yє
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/add_1к
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
М▒
─
!__inference__traced_restore_64088
file_prefix/
+assignvariableop_model_ecc_conv_root_kernel,
(assignvariableop_1_model_gcn_conv_kernel.
*assignvariableop_2_model_gcn_conv_1_kernel.
*assignvariableop_3_model_gcn_conv_2_kernel.
*assignvariableop_4_model_gcn_conv_3_kernel+
'assignvariableop_5_model_dense_5_kernel)
%assignvariableop_6_model_dense_5_bias2
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
&assignvariableop_21_model_dense_4_bias7
3assignvariableop_22_model_batch_normalization_gamma6
2assignvariableop_23_model_batch_normalization_beta9
5assignvariableop_24_model_batch_normalization_1_gamma8
4assignvariableop_25_model_batch_normalization_1_beta9
5assignvariableop_26_model_batch_normalization_2_gamma8
4assignvariableop_27_model_batch_normalization_2_beta9
5assignvariableop_28_model_batch_normalization_3_gamma8
4assignvariableop_29_model_batch_normalization_3_beta9
5assignvariableop_30_model_batch_normalization_4_gamma8
4assignvariableop_31_model_batch_normalization_4_beta=
9assignvariableop_32_model_batch_normalization_moving_meanA
=assignvariableop_33_model_batch_normalization_moving_variance?
;assignvariableop_34_model_batch_normalization_1_moving_meanC
?assignvariableop_35_model_batch_normalization_1_moving_variance?
;assignvariableop_36_model_batch_normalization_2_moving_meanC
?assignvariableop_37_model_batch_normalization_2_moving_variance?
;assignvariableop_38_model_batch_normalization_3_moving_meanC
?assignvariableop_39_model_batch_normalization_3_moving_variance?
;assignvariableop_40_model_batch_normalization_4_moving_meanC
?assignvariableop_41_model_batch_normalization_4_moving_variance
identity_43ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9╬
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*┌
valueлB═+B+ECC1/root_kernel/.ATTRIBUTES/VARIABLE_VALUEB&GCN1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&GCN2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&GCN3/kernel/.ATTRIBUTES/VARIABLE_VALUEB&GCN4/kernel/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesС
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЁ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*┬
_output_shapes»
г:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityф
AssignVariableOpAssignVariableOp+assignvariableop_model_ecc_conv_root_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Г
AssignVariableOp_1AssignVariableOp(assignvariableop_1_model_gcn_conv_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2»
AssignVariableOp_2AssignVariableOp*assignvariableop_2_model_gcn_conv_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3»
AssignVariableOp_3AssignVariableOp*assignvariableop_3_model_gcn_conv_2_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4»
AssignVariableOp_4AssignVariableOp*assignvariableop_4_model_gcn_conv_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5г
AssignVariableOp_5AssignVariableOp'assignvariableop_5_model_dense_5_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6ф
AssignVariableOp_6AssignVariableOp%assignvariableop_6_model_dense_5_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7│
AssignVariableOp_7AssignVariableOp.assignvariableop_7_model_ecc_conv_fgn_0_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8│
AssignVariableOp_8AssignVariableOp.assignvariableop_8_model_ecc_conv_fgn_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9│
AssignVariableOp_9AssignVariableOp.assignvariableop_9_model_ecc_conv_fgn_2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10╣
AssignVariableOp_10AssignVariableOp1assignvariableop_10_model_ecc_conv_fgn_out_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11и
AssignVariableOp_11AssignVariableOp/assignvariableop_11_model_ecc_conv_fgn_out_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp&assignvariableop_12_model_dense_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13г
AssignVariableOp_13AssignVariableOp$assignvariableop_13_model_dense_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14░
AssignVariableOp_14AssignVariableOp(assignvariableop_14_model_dense_1_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15«
AssignVariableOp_15AssignVariableOp&assignvariableop_15_model_dense_1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16░
AssignVariableOp_16AssignVariableOp(assignvariableop_16_model_dense_2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17«
AssignVariableOp_17AssignVariableOp&assignvariableop_17_model_dense_2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18░
AssignVariableOp_18AssignVariableOp(assignvariableop_18_model_dense_3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19«
AssignVariableOp_19AssignVariableOp&assignvariableop_19_model_dense_3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20░
AssignVariableOp_20AssignVariableOp(assignvariableop_20_model_dense_4_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21«
AssignVariableOp_21AssignVariableOp&assignvariableop_21_model_dense_4_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22╗
AssignVariableOp_22AssignVariableOp3assignvariableop_22_model_batch_normalization_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23║
AssignVariableOp_23AssignVariableOp2assignvariableop_23_model_batch_normalization_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24й
AssignVariableOp_24AssignVariableOp5assignvariableop_24_model_batch_normalization_1_gammaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25╝
AssignVariableOp_25AssignVariableOp4assignvariableop_25_model_batch_normalization_1_betaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26й
AssignVariableOp_26AssignVariableOp5assignvariableop_26_model_batch_normalization_2_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27╝
AssignVariableOp_27AssignVariableOp4assignvariableop_27_model_batch_normalization_2_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28й
AssignVariableOp_28AssignVariableOp5assignvariableop_28_model_batch_normalization_3_gammaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29╝
AssignVariableOp_29AssignVariableOp4assignvariableop_29_model_batch_normalization_3_betaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30й
AssignVariableOp_30AssignVariableOp5assignvariableop_30_model_batch_normalization_4_gammaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31╝
AssignVariableOp_31AssignVariableOp4assignvariableop_31_model_batch_normalization_4_betaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32┴
AssignVariableOp_32AssignVariableOp9assignvariableop_32_model_batch_normalization_moving_meanIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33┼
AssignVariableOp_33AssignVariableOp=assignvariableop_33_model_batch_normalization_moving_varianceIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34├
AssignVariableOp_34AssignVariableOp;assignvariableop_34_model_batch_normalization_1_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35К
AssignVariableOp_35AssignVariableOp?assignvariableop_35_model_batch_normalization_1_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36├
AssignVariableOp_36AssignVariableOp;assignvariableop_36_model_batch_normalization_2_moving_meanIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37К
AssignVariableOp_37AssignVariableOp?assignvariableop_37_model_batch_normalization_2_moving_varianceIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38├
AssignVariableOp_38AssignVariableOp;assignvariableop_38_model_batch_normalization_3_moving_meanIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39К
AssignVariableOp_39AssignVariableOp?assignvariableop_39_model_batch_normalization_3_moving_varianceIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40├
AssignVariableOp_40AssignVariableOp;assignvariableop_40_model_batch_normalization_4_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41К
AssignVariableOp_41AssignVariableOp?assignvariableop_41_model_batch_normalization_4_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_419
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЩ
Identity_42Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_42ь
Identity_43IdentityIdentity_42:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_43"#
identity_43Identity_43:output:0*┐
_input_shapesГ
ф: ::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412(
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
ў
█
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_63773

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identityѕбCast/ReadVariableOpбCast_1/ReadVariableOpбCast_2/ReadVariableOpбCast_3/ReadVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_1/ReadVariableOpі
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_2/ReadVariableOpі
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yє
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/add_1к
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
П
z
%__inference_dense_layer_call_fn_63313

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_612552
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
с
Ф
%__inference_model_layer_call_fn_62979
inputs_0

inputs	
inputs_1
inputs_2	

inputs_2_0	
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

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identityѕбStatefulPartitionedCall└
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2
inputs_2_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*:
Tin3
12/			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *B
_read_only_resource_inputs$
" 	
 !"%&'(+,-.*2
config_proto" 

CPU

GPU2 *0J 8ѓ *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_618972
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Є
_input_shapesш
Ы:         :         :         ::         ::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:MI
#
_output_shapes
:         
"
_user_specified_name
inputs/2
б/
Г
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_63671

inputs
assignmovingavg_63646
assignmovingavg_1_63652 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб AssignMovingAvg_1/ReadVariableOpбCast/ReadVariableOpбCast_1/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesљ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ђ2
moments/StopGradientЦ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         ђ2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeeze_1╦
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/63646*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayЊ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_63646*
_output_shapes	
:ђ*
dtype02 
AssignMovingAvg/ReadVariableOpы
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/63646*
_output_shapes	
:ђ2
AssignMovingAvg/subУ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/63646*
_output_shapes	
:ђ2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_63646AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/63646*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЛ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/63652*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayЎ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_63652*
_output_shapes	
:ђ*
dtype02"
 AssignMovingAvg_1/ReadVariableOpч
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/63652*
_output_shapes	
:ђ2
AssignMovingAvg_1/subЫ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/63652*
_output_shapes	
:ђ2
AssignMovingAvg_1/mul╣
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_63652AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/63652*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/add_1е
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
б/
Г
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_63753

inputs
assignmovingavg_63728
assignmovingavg_1_63734 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб AssignMovingAvg_1/ReadVariableOpбCast/ReadVariableOpбCast_1/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesљ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ђ2
moments/StopGradientЦ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         ђ2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeeze_1╦
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/63728*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayЊ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_63728*
_output_shapes	
:ђ*
dtype02 
AssignMovingAvg/ReadVariableOpы
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/63728*
_output_shapes	
:ђ2
AssignMovingAvg/subУ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/63728*
_output_shapes	
:ђ2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_63728AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/63728*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЛ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/63734*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayЎ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_63734*
_output_shapes	
:ђ*
dtype02"
 AssignMovingAvg_1/ReadVariableOpч
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/63734*
_output_shapes	
:ђ2
AssignMovingAvg_1/subЫ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/63734*
_output_shapes	
:ђ2
AssignMovingAvg_1/mul╣
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_63734AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/63734*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/add_1е
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ц
[
/__inference_global_max_pool_layer_call_fn_63251
inputs_0
inputs_1	
identity█
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_global_max_pool_layer_call_and_return_conditional_losses_612042
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ђ:         :R N
(
_output_shapes
:         ђ
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:         
"
_user_specified_name
inputs/1
ў
█
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_60521

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identityѕбCast/ReadVariableOpбCast_1/ReadVariableOpбCast_2/ReadVariableOpбCast_3/ReadVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_1/ReadVariableOpі
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_2/ReadVariableOpі
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yє
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/add_1к
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ў	
█
B__inference_dense_4_layer_call_and_return_conditional_losses_61503

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddќ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ЃЮ
Й
@__inference_model_layer_call_and_return_conditional_losses_62144

inputs
inputs_1	
inputs_2
inputs_3	
inputs_4	
ecc_conv_62033
ecc_conv_62035
ecc_conv_62037
ecc_conv_62039
ecc_conv_62041
ecc_conv_62043
gcn_conv_62046
gcn_conv_1_62049
gcn_conv_2_62052
gcn_conv_3_62055
dense_62063
dense_62065
batch_normalization_62069
batch_normalization_62071
batch_normalization_62073
batch_normalization_62075
dense_1_62078
dense_1_62080
batch_normalization_1_62084
batch_normalization_1_62086
batch_normalization_1_62088
batch_normalization_1_62090
dense_2_62093
dense_2_62095
batch_normalization_2_62099
batch_normalization_2_62101
batch_normalization_2_62103
batch_normalization_2_62105
dense_3_62108
dense_3_62110
batch_normalization_3_62114
batch_normalization_3_62116
batch_normalization_3_62118
batch_normalization_3_62120
dense_4_62123
dense_4_62125
batch_normalization_4_62129
batch_normalization_4_62131
batch_normalization_4_62133
batch_normalization_4_62135
dense_5_62138
dense_5_62140
identityѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallб-batch_normalization_3/StatefulPartitionedCallб-batch_normalization_4/StatefulPartitionedCallбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбdense_3/StatefulPartitionedCallбdense_4/StatefulPartitionedCallбdense_5/StatefulPartitionedCallб ecc_conv/StatefulPartitionedCallб gcn_conv/StatefulPartitionedCallб"gcn_conv_1/StatefulPartitionedCallб"gcn_conv_2/StatefulPartitionedCallб"gcn_conv_3/StatefulPartitionedCall{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2І
strided_sliceStridedSliceinputs_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stackЃ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1Ѓ
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2Ћ
strided_slice_1StridedSliceinputs_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis»
GatherV2GatherV2inputsstrided_slice_1:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:         2

GatherV2d
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_1/axis│

GatherV2_1GatherV2inputsstrided_slice:output:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:         2

GatherV2_1k
SubSubGatherV2:output:0GatherV2_1:output:0*
T0*'
_output_shapes
:         2
Sub
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stackЃ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1Ѓ
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2ђ
strided_slice_2StridedSliceSub:z:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
strided_slice_2f
SquareSquarestrided_slice_2:output:0*
T0*'
_output_shapes
:         2
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesk
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
SumP
SqrtSqrtSum:output:0*
T0*#
_output_shapes
:         2
Sqrt
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stackЃ
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1Ѓ
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2ђ
strided_slice_3StridedSliceSub:z:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
strided_slice_3k
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         2
ExpandDims/dim{

ExpandDims
ExpandDimsSqrt:y:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:         2

ExpandDimsЁ

div_no_nanDivNoNanstrided_slice_3:output:0ExpandDims:output:0*
T0*'
_output_shapes
:         2

div_no_nan
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stackЃ
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_4/stack_1Ѓ
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2ђ
strided_slice_4StridedSliceSub:z:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
strided_slice_4o
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
         2
ExpandDims_1/dimЂ
ExpandDims_1
ExpandDimsSqrt:y:0ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:         2
ExpandDims_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis«
concatConcatV2strided_slice_4:output:0ExpandDims_1:output:0div_no_nan:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
concatЉ
 ecc_conv/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3concat:output:0ecc_conv_62033ecc_conv_62035ecc_conv_62037ecc_conv_62039ecc_conv_62041ecc_conv_62043*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*(
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *L
fGRE
C__inference_ecc_conv_layer_call_and_return_conditional_losses_610542"
 ecc_conv/StatefulPartitionedCall╚
 gcn_conv/StatefulPartitionedCallStatefulPartitionedCall)ecc_conv/StatefulPartitionedCall:output:0inputs_1inputs_2inputs_3gcn_conv_62046*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *L
fGRE
C__inference_gcn_conv_layer_call_and_return_conditional_losses_611022"
 gcn_conv/StatefulPartitionedCallЛ
"gcn_conv_1/StatefulPartitionedCallStatefulPartitionedCall)gcn_conv/StatefulPartitionedCall:output:0inputs_1inputs_2inputs_3gcn_conv_1_62049*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_gcn_conv_1_layer_call_and_return_conditional_losses_611292$
"gcn_conv_1/StatefulPartitionedCallМ
"gcn_conv_2/StatefulPartitionedCallStatefulPartitionedCall+gcn_conv_1/StatefulPartitionedCall:output:0inputs_1inputs_2inputs_3gcn_conv_2_62052*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_gcn_conv_2_layer_call_and_return_conditional_losses_611562$
"gcn_conv_2/StatefulPartitionedCallМ
"gcn_conv_3/StatefulPartitionedCallStatefulPartitionedCall+gcn_conv_2/StatefulPartitionedCall:output:0inputs_1inputs_2inputs_3gcn_conv_3_62055*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_gcn_conv_3_layer_call_and_return_conditional_losses_611832$
"gcn_conv_3/StatefulPartitionedCallъ
global_max_pool/PartitionedCallPartitionedCall+gcn_conv_3/StatefulPartitionedCall:output:0inputs_4*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_global_max_pool_layer_call_and_return_conditional_losses_612042!
global_max_pool/PartitionedCallъ
global_avg_pool/PartitionedCallPartitionedCall+gcn_conv_3/StatefulPartitionedCall:output:0inputs_4*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_global_avg_pool_layer_call_and_return_conditional_losses_612192!
global_avg_pool/PartitionedCallъ
global_sum_pool/PartitionedCallPartitionedCall+gcn_conv_3/StatefulPartitionedCall:output:0inputs_4*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_global_sum_pool_layer_call_and_return_conditional_losses_612342!
global_sum_pool/PartitionedCall`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axisЫ
concat_1ConcatV2(global_max_pool/PartitionedCall:output:0(global_avg_pool/PartitionedCall:output:0(global_sum_pool/PartitionedCall:output:0concat_1/axis:output:0*
N*
T0*(
_output_shapes
:         ђ2

concat_1Њ
dense/StatefulPartitionedCallStatefulPartitionedCallconcat_1:output:0dense_62063dense_62065*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_612552
dense/StatefulPartitionedCallЮ
leaky_re_lu/LeakyRelu	LeakyRelu&dense/StatefulPartitionedCall:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
leaky_re_lu/LeakyReluЦ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall#leaky_re_lu/LeakyRelu:activations:0batch_normalization_62069batch_normalization_62071batch_normalization_62073batch_normalization_62075*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_603812-
+batch_normalization/StatefulPartitionedCall└
dense_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_1_62078dense_1_62080*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_613172!
dense_1/StatefulPartitionedCallБ
leaky_re_lu/LeakyRelu_1	LeakyRelu(dense_1/StatefulPartitionedCall:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
leaky_re_lu/LeakyRelu_1х
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall%leaky_re_lu/LeakyRelu_1:activations:0batch_normalization_1_62084batch_normalization_1_62086batch_normalization_1_62088batch_normalization_1_62090*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_605212/
-batch_normalization_1/StatefulPartitionedCall┬
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_2_62093dense_2_62095*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_613792!
dense_2/StatefulPartitionedCallБ
leaky_re_lu/LeakyRelu_2	LeakyRelu(dense_2/StatefulPartitionedCall:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
leaky_re_lu/LeakyRelu_2х
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_re_lu/LeakyRelu_2:activations:0batch_normalization_2_62099batch_normalization_2_62101batch_normalization_2_62103batch_normalization_2_62105*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_606612/
-batch_normalization_2/StatefulPartitionedCall┬
dense_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_3_62108dense_3_62110*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_614412!
dense_3/StatefulPartitionedCallБ
leaky_re_lu/LeakyRelu_3	LeakyRelu(dense_3/StatefulPartitionedCall:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
leaky_re_lu/LeakyRelu_3х
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall%leaky_re_lu/LeakyRelu_3:activations:0batch_normalization_3_62114batch_normalization_3_62116batch_normalization_3_62118batch_normalization_3_62120*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_608012/
-batch_normalization_3/StatefulPartitionedCall┬
dense_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_4_62123dense_4_62125*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_615032!
dense_4/StatefulPartitionedCallБ
leaky_re_lu/LeakyRelu_4	LeakyRelu(dense_4/StatefulPartitionedCall:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
leaky_re_lu/LeakyRelu_4х
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall%leaky_re_lu/LeakyRelu_4:activations:0batch_normalization_4_62129batch_normalization_4_62131batch_normalization_4_62133batch_normalization_4_62135*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_609412/
-batch_normalization_4/StatefulPartitionedCall┴
dense_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_5_62138dense_5_62140*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_615652!
dense_5/StatefulPartitionedCallж
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall!^ecc_conv/StatefulPartitionedCall!^gcn_conv/StatefulPartitionedCall#^gcn_conv_1/StatefulPartitionedCall#^gcn_conv_2/StatefulPartitionedCall#^gcn_conv_3/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Є
_input_shapesш
Ы:         :         :         ::         ::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2D
 ecc_conv/StatefulPartitionedCall ecc_conv/StatefulPartitionedCall2D
 gcn_conv/StatefulPartitionedCall gcn_conv/StatefulPartitionedCall2H
"gcn_conv_1/StatefulPartitionedCall"gcn_conv_1/StatefulPartitionedCall2H
"gcn_conv_2/StatefulPartitionedCall"gcn_conv_2/StatefulPartitionedCall2H
"gcn_conv_3/StatefulPartitionedCall"gcn_conv_3/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs
а/
Ф
N__inference_batch_normalization_layer_call_and_return_conditional_losses_60348

inputs
assignmovingavg_60323
assignmovingavg_1_60329 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб AssignMovingAvg_1/ReadVariableOpбCast/ReadVariableOpбCast_1/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesљ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ђ2
moments/StopGradientЦ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         ђ2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeeze_1╦
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/60323*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayЊ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_60323*
_output_shapes	
:ђ*
dtype02 
AssignMovingAvg/ReadVariableOpы
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/60323*
_output_shapes	
:ђ2
AssignMovingAvg/subУ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/60323*
_output_shapes	
:ђ2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_60323AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/60323*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЛ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/60329*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayЎ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_60329*
_output_shapes	
:ђ*
dtype02"
 AssignMovingAvg_1/ReadVariableOpч
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/60329*
_output_shapes	
:ђ2
AssignMovingAvg_1/subЫ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/60329*
_output_shapes	
:ђ2
AssignMovingAvg_1/mul╣
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_60329AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/60329*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/add_1е
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ў
█
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_60801

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identityѕбCast/ReadVariableOpбCast_1/ReadVariableOpбCast_2/ReadVariableOpбCast_3/ReadVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_1/ReadVariableOpі
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_2/ReadVariableOpі
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yє
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/add_1к
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
а/
Ф
N__inference_batch_normalization_layer_call_and_return_conditional_losses_63425

inputs
assignmovingavg_63400
assignmovingavg_1_63406 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб AssignMovingAvg_1/ReadVariableOpбCast/ReadVariableOpбCast_1/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesљ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ђ2
moments/StopGradientЦ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         ђ2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeeze_1╦
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/63400*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayЊ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_63400*
_output_shapes	
:ђ*
dtype02 
AssignMovingAvg/ReadVariableOpы
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/63400*
_output_shapes	
:ђ2
AssignMovingAvg/subУ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/63400*
_output_shapes	
:ђ2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_63400AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/63400*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЛ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/63406*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayЎ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_63406*
_output_shapes	
:ђ*
dtype02"
 AssignMovingAvg_1/ReadVariableOpч
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/63406*
_output_shapes	
:ђ2
AssignMovingAvg_1/subЫ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/63406*
_output_shapes	
:ђ2
AssignMovingAvg_1/mul╣
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_63406AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/63406*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/add_1е
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ќ
╩
E__inference_gcn_conv_2_layer_call_and_return_conditional_losses_61156

inputs
inputs_1	
inputs_2
inputs_3	"
matmul_readvariableop_resource
identityѕбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЯ
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1inputs_2inputs_3MatMul:product:0*
T0*(
_output_shapes
:         ђ21
/SparseTensorDenseMatMul/SparseTensorDenseMatMulѓ
ReluRelu9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:         ђ2
Relu
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*S
_input_shapesB
@:         ђ:         :         ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
Я
v
J__inference_global_avg_pool_layer_call_and_return_conditional_losses_63257
inputs_0
inputs_1	
identityђ
SegmentMeanSegmentMeaninputs_0inputs_1*
T0*
Tindices0	*(
_output_shapes
:         ђ2
SegmentMeani
IdentityIdentitySegmentMean:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ђ:         :R N
(
_output_shapes
:         ђ
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:         
"
_user_specified_name
inputs/1
Й
е
5__inference_batch_normalization_1_layer_call_fn_63553

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_605212
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ў
█
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_60941

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identityѕбCast/ReadVariableOpбCast_1/ReadVariableOpбCast_2/ReadVariableOpбCast_3/ReadVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_1/ReadVariableOpі
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_2/ReadVariableOpі
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yє
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/add_1к
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ќ	
┘
@__inference_dense_layer_call_and_return_conditional_losses_61255

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddќ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Й
е
5__inference_batch_normalization_3_layer_call_fn_63717

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_608012
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ў
╩
E__inference_gcn_conv_2_layer_call_and_return_conditional_losses_63207
inputs_0

inputs	
inputs_1
inputs_2	"
matmul_readvariableop_resource
identityѕбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpv
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulя
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2MatMul:product:0*
T0*(
_output_shapes
:         ђ21
/SparseTensorDenseMatMul/SparseTensorDenseMatMulѓ
ReluRelu9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:         ђ2
Relu
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*S
_input_shapesB
@:         ђ:         :         ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:R N
(
_output_shapes
:         ђ
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
ў
█
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_60661

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identityѕбCast/ReadVariableOpбCast_1/ReadVariableOpбCast_2/ReadVariableOpбCast_3/ReadVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_1/ReadVariableOpі
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_2/ReadVariableOpі
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yє
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/add_1к
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Њ	
█
B__inference_dense_5_layer_call_and_return_conditional_losses_61565

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddЋ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
б/
Г
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_63507

inputs
assignmovingavg_63482
assignmovingavg_1_63488 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб AssignMovingAvg_1/ReadVariableOpбCast/ReadVariableOpбCast_1/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesљ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ђ2
moments/StopGradientЦ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         ђ2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeeze_1╦
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/63482*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayЊ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_63482*
_output_shapes	
:ђ*
dtype02 
AssignMovingAvg/ReadVariableOpы
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/63482*
_output_shapes	
:ђ2
AssignMovingAvg/subУ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/63482*
_output_shapes	
:ђ2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_63482AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/63482*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЛ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/63488*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayЎ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_63488*
_output_shapes	
:ђ*
dtype02"
 AssignMovingAvg_1/ReadVariableOpч
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/63488*
_output_shapes	
:ђ2
AssignMovingAvg_1/subЫ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/63488*
_output_shapes	
:ђ2
AssignMovingAvg_1/mul╣
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_63488AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/63488*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/add_1е
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
М
t
J__inference_global_max_pool_layer_call_and_return_conditional_losses_61204

inputs
inputs_1	
identity{

SegmentMax
SegmentMaxinputsinputs_1*
T0*
Tindices0	*(
_output_shapes
:         ђ2

SegmentMaxh
IdentityIdentitySegmentMax:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ђ:         :P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs
ў
╩
E__inference_gcn_conv_3_layer_call_and_return_conditional_losses_63229
inputs_0

inputs	
inputs_1
inputs_2	"
matmul_readvariableop_resource
identityѕбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpv
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulя
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2MatMul:product:0*
T0*(
_output_shapes
:         ђ21
/SparseTensorDenseMatMul/SparseTensorDenseMatMulѓ
ReluRelu9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*(
_output_shapes
:         ђ2
Relu
IdentityIdentityRelu:activations:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*S
_input_shapesB
@:         ђ:         :         ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:R N
(
_output_shapes
:         ђ
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
О
t
J__inference_global_avg_pool_layer_call_and_return_conditional_losses_61219

inputs
inputs_1	
identity~
SegmentMeanSegmentMeaninputsinputs_1*
T0*
Tindices0	*(
_output_shapes
:         ђ2
SegmentMeani
IdentityIdentitySegmentMean:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ђ:         :P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs
б/
Г
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_60488

inputs
assignmovingavg_60463
assignmovingavg_1_60469 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб AssignMovingAvg_1/ReadVariableOpбCast/ReadVariableOpбCast_1/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesљ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ђ2
moments/StopGradientЦ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         ђ2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђ*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђ*
squeeze_dims
 2
moments/Squeeze_1╦
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/60463*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayЊ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_60463*
_output_shapes	
:ђ*
dtype02 
AssignMovingAvg/ReadVariableOpы
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/60463*
_output_shapes	
:ђ2
AssignMovingAvg/subУ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/60463*
_output_shapes	
:ђ2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_60463AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/60463*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЛ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/60469*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayЎ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_60469*
_output_shapes	
:ђ*
dtype02"
 AssignMovingAvg_1/ReadVariableOpч
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/60469*
_output_shapes	
:ђ2
AssignMovingAvg_1/subЫ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/60469*
_output_shapes	
:ђ2
AssignMovingAvg_1/mul╣
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_60469AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/60469*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpё
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast/ReadVariableOpі
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђ2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђ2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ2
batchnorm/add_1е
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
щю
Й
@__inference_model_layer_call_and_return_conditional_losses_61897

inputs
inputs_1	
inputs_2
inputs_3	
inputs_4	
ecc_conv_61786
ecc_conv_61788
ecc_conv_61790
ecc_conv_61792
ecc_conv_61794
ecc_conv_61796
gcn_conv_61799
gcn_conv_1_61802
gcn_conv_2_61805
gcn_conv_3_61808
dense_61816
dense_61818
batch_normalization_61822
batch_normalization_61824
batch_normalization_61826
batch_normalization_61828
dense_1_61831
dense_1_61833
batch_normalization_1_61837
batch_normalization_1_61839
batch_normalization_1_61841
batch_normalization_1_61843
dense_2_61846
dense_2_61848
batch_normalization_2_61852
batch_normalization_2_61854
batch_normalization_2_61856
batch_normalization_2_61858
dense_3_61861
dense_3_61863
batch_normalization_3_61867
batch_normalization_3_61869
batch_normalization_3_61871
batch_normalization_3_61873
dense_4_61876
dense_4_61878
batch_normalization_4_61882
batch_normalization_4_61884
batch_normalization_4_61886
batch_normalization_4_61888
dense_5_61891
dense_5_61893
identityѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallб-batch_normalization_3/StatefulPartitionedCallб-batch_normalization_4/StatefulPartitionedCallбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбdense_3/StatefulPartitionedCallбdense_4/StatefulPartitionedCallбdense_5/StatefulPartitionedCallб ecc_conv/StatefulPartitionedCallб gcn_conv/StatefulPartitionedCallб"gcn_conv_1/StatefulPartitionedCallб"gcn_conv_2/StatefulPartitionedCallб"gcn_conv_3/StatefulPartitionedCall{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2І
strided_sliceStridedSliceinputs_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stackЃ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1Ѓ
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2Ћ
strided_slice_1StridedSliceinputs_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis»
GatherV2GatherV2inputsstrided_slice_1:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:         2

GatherV2d
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_1/axis│

GatherV2_1GatherV2inputsstrided_slice:output:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:         2

GatherV2_1k
SubSubGatherV2:output:0GatherV2_1:output:0*
T0*'
_output_shapes
:         2
Sub
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stackЃ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1Ѓ
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2ђ
strided_slice_2StridedSliceSub:z:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
strided_slice_2f
SquareSquarestrided_slice_2:output:0*
T0*'
_output_shapes
:         2
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesk
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         2
SumP
SqrtSqrtSum:output:0*
T0*#
_output_shapes
:         2
Sqrt
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stackЃ
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1Ѓ
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2ђ
strided_slice_3StridedSliceSub:z:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
strided_slice_3k
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         2
ExpandDims/dim{

ExpandDims
ExpandDimsSqrt:y:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:         2

ExpandDimsЁ

div_no_nanDivNoNanstrided_slice_3:output:0ExpandDims:output:0*
T0*'
_output_shapes
:         2

div_no_nan
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stackЃ
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_4/stack_1Ѓ
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2ђ
strided_slice_4StridedSliceSub:z:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
strided_slice_4o
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
         2
ExpandDims_1/dimЂ
ExpandDims_1
ExpandDimsSqrt:y:0ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:         2
ExpandDims_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis«
concatConcatV2strided_slice_4:output:0ExpandDims_1:output:0div_no_nan:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
concatЉ
 ecc_conv/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3concat:output:0ecc_conv_61786ecc_conv_61788ecc_conv_61790ecc_conv_61792ecc_conv_61794ecc_conv_61796*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*(
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *L
fGRE
C__inference_ecc_conv_layer_call_and_return_conditional_losses_610542"
 ecc_conv/StatefulPartitionedCall╚
 gcn_conv/StatefulPartitionedCallStatefulPartitionedCall)ecc_conv/StatefulPartitionedCall:output:0inputs_1inputs_2inputs_3gcn_conv_61799*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *L
fGRE
C__inference_gcn_conv_layer_call_and_return_conditional_losses_611022"
 gcn_conv/StatefulPartitionedCallЛ
"gcn_conv_1/StatefulPartitionedCallStatefulPartitionedCall)gcn_conv/StatefulPartitionedCall:output:0inputs_1inputs_2inputs_3gcn_conv_1_61802*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_gcn_conv_1_layer_call_and_return_conditional_losses_611292$
"gcn_conv_1/StatefulPartitionedCallМ
"gcn_conv_2/StatefulPartitionedCallStatefulPartitionedCall+gcn_conv_1/StatefulPartitionedCall:output:0inputs_1inputs_2inputs_3gcn_conv_2_61805*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_gcn_conv_2_layer_call_and_return_conditional_losses_611562$
"gcn_conv_2/StatefulPartitionedCallМ
"gcn_conv_3/StatefulPartitionedCallStatefulPartitionedCall+gcn_conv_2/StatefulPartitionedCall:output:0inputs_1inputs_2inputs_3gcn_conv_3_61808*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_gcn_conv_3_layer_call_and_return_conditional_losses_611832$
"gcn_conv_3/StatefulPartitionedCallъ
global_max_pool/PartitionedCallPartitionedCall+gcn_conv_3/StatefulPartitionedCall:output:0inputs_4*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_global_max_pool_layer_call_and_return_conditional_losses_612042!
global_max_pool/PartitionedCallъ
global_avg_pool/PartitionedCallPartitionedCall+gcn_conv_3/StatefulPartitionedCall:output:0inputs_4*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_global_avg_pool_layer_call_and_return_conditional_losses_612192!
global_avg_pool/PartitionedCallъ
global_sum_pool/PartitionedCallPartitionedCall+gcn_conv_3/StatefulPartitionedCall:output:0inputs_4*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_global_sum_pool_layer_call_and_return_conditional_losses_612342!
global_sum_pool/PartitionedCall`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axisЫ
concat_1ConcatV2(global_max_pool/PartitionedCall:output:0(global_avg_pool/PartitionedCall:output:0(global_sum_pool/PartitionedCall:output:0concat_1/axis:output:0*
N*
T0*(
_output_shapes
:         ђ2

concat_1Њ
dense/StatefulPartitionedCallStatefulPartitionedCallconcat_1:output:0dense_61816dense_61818*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_612552
dense/StatefulPartitionedCallЮ
leaky_re_lu/LeakyRelu	LeakyRelu&dense/StatefulPartitionedCall:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
leaky_re_lu/LeakyReluБ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall#leaky_re_lu/LeakyRelu:activations:0batch_normalization_61822batch_normalization_61824batch_normalization_61826batch_normalization_61828*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_603482-
+batch_normalization/StatefulPartitionedCall└
dense_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_1_61831dense_1_61833*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_613172!
dense_1/StatefulPartitionedCallБ
leaky_re_lu/LeakyRelu_1	LeakyRelu(dense_1/StatefulPartitionedCall:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
leaky_re_lu/LeakyRelu_1│
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall%leaky_re_lu/LeakyRelu_1:activations:0batch_normalization_1_61837batch_normalization_1_61839batch_normalization_1_61841batch_normalization_1_61843*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_604882/
-batch_normalization_1/StatefulPartitionedCall┬
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_2_61846dense_2_61848*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_613792!
dense_2/StatefulPartitionedCallБ
leaky_re_lu/LeakyRelu_2	LeakyRelu(dense_2/StatefulPartitionedCall:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
leaky_re_lu/LeakyRelu_2│
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_re_lu/LeakyRelu_2:activations:0batch_normalization_2_61852batch_normalization_2_61854batch_normalization_2_61856batch_normalization_2_61858*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_606282/
-batch_normalization_2/StatefulPartitionedCall┬
dense_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_3_61861dense_3_61863*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_614412!
dense_3/StatefulPartitionedCallБ
leaky_re_lu/LeakyRelu_3	LeakyRelu(dense_3/StatefulPartitionedCall:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
leaky_re_lu/LeakyRelu_3│
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall%leaky_re_lu/LeakyRelu_3:activations:0batch_normalization_3_61867batch_normalization_3_61869batch_normalization_3_61871batch_normalization_3_61873*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_607682/
-batch_normalization_3/StatefulPartitionedCall┬
dense_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_4_61876dense_4_61878*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_615032!
dense_4/StatefulPartitionedCallБ
leaky_re_lu/LeakyRelu_4	LeakyRelu(dense_4/StatefulPartitionedCall:output:0*(
_output_shapes
:         ђ*
alpha%═╠╠=2
leaky_re_lu/LeakyRelu_4│
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall%leaky_re_lu/LeakyRelu_4:activations:0batch_normalization_4_61882batch_normalization_4_61884batch_normalization_4_61886batch_normalization_4_61888*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_609082/
-batch_normalization_4/StatefulPartitionedCall┴
dense_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_5_61891dense_5_61893*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_615652!
dense_5/StatefulPartitionedCallж
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall!^ecc_conv/StatefulPartitionedCall!^gcn_conv/StatefulPartitionedCall#^gcn_conv_1/StatefulPartitionedCall#^gcn_conv_2/StatefulPartitionedCall#^gcn_conv_3/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Є
_input_shapesш
Ы:         :         :         ::         ::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2D
 ecc_conv/StatefulPartitionedCall ecc_conv/StatefulPartitionedCall2D
 gcn_conv/StatefulPartitionedCall gcn_conv/StatefulPartitionedCall2H
"gcn_conv_1/StatefulPartitionedCall"gcn_conv_1/StatefulPartitionedCall2H
"gcn_conv_2/StatefulPartitionedCall"gcn_conv_2/StatefulPartitionedCall2H
"gcn_conv_3/StatefulPartitionedCall"gcn_conv_3/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs:KG
#
_output_shapes
:         
 
_user_specified_nameinputs
╝
е
5__inference_batch_normalization_4_layer_call_fn_63786

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_609082
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs"▒L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*љ
serving_defaultЧ
9
args_0/
serving_default_args_0:0         
=
args_0_11
serving_default_args_0_1:0	         
9
args_0_2-
serving_default_args_0_2:0         
0
args_0_3$
serving_default_args_0_3:0	
9
args_0_4-
serving_default_args_0_4:0	         <
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:ЦЗ
Д
ECC1
GCN1
GCN2
GCN3
GCN4
	Pool1
	Pool2
	Pool3

	decode

norm_layers
d2
regularization_losses
trainable_variables
	variables
	keras_api

signatures
+Е&call_and_return_all_conditional_losses
ф__call__
Ф_default_save_signature"­
_tf_keras_modelо{"class_name": "model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "model"}}
ќ
kwargs_keys
kernel_network
kernel_network_layers
root_kernel
regularization_losses
trainable_variables
	variables
	keras_api
+г&call_and_return_all_conditional_losses
Г__call__"┤
_tf_keras_layerџ{"class_name": "ECCConv", "name": "ecc_conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ecc_conv", "trainable": true, "dtype": "float32", "activation": "relu", "use_bias": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "channels": 64, "kernel_network": [64, 64, 64], "root": true}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 5]}, {"class_name": "TensorShape", "items": [null, null]}, {"class_name": "TensorShape", "items": [null, 6]}]}
ѓ
kwargs_keys

kernel
regularization_losses
trainable_variables
	variables
	keras_api
+«&call_and_return_all_conditional_losses
»__call__"н
_tf_keras_layer║{"class_name": "GCNConv", "name": "gcn_conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gcn_conv", "trainable": true, "dtype": "float32", "activation": "relu", "use_bias": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "channels": 64}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 64]}, {"class_name": "TensorShape", "items": [null, null]}]}
Є
kwargs_keys

 kernel
!regularization_losses
"trainable_variables
#	variables
$	keras_api
+░&call_and_return_all_conditional_losses
▒__call__"┘
_tf_keras_layer┐{"class_name": "GCNConv", "name": "gcn_conv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gcn_conv_1", "trainable": true, "dtype": "float32", "activation": "relu", "use_bias": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "channels": 128}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 64]}, {"class_name": "TensorShape", "items": [null, null]}]}
ѕ
%kwargs_keys

&kernel
'regularization_losses
(trainable_variables
)	variables
*	keras_api
+▓&call_and_return_all_conditional_losses
│__call__"┌
_tf_keras_layer└{"class_name": "GCNConv", "name": "gcn_conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gcn_conv_2", "trainable": true, "dtype": "float32", "activation": "relu", "use_bias": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "channels": 256}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 128]}, {"class_name": "TensorShape", "items": [null, null]}]}
ѕ
+kwargs_keys

,kernel
-regularization_losses
.trainable_variables
/	variables
0	keras_api
+┤&call_and_return_all_conditional_losses
х__call__"┌
_tf_keras_layer└{"class_name": "GCNConv", "name": "gcn_conv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gcn_conv_3", "trainable": true, "dtype": "float32", "activation": "relu", "use_bias": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "channels": 512}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 256]}, {"class_name": "TensorShape", "items": [null, null]}]}
к
1regularization_losses
2trainable_variables
3	variables
4	keras_api
+Х&call_and_return_all_conditional_losses
и__call__"х
_tf_keras_layerЏ{"class_name": "GlobalMaxPool", "name": "global_max_pool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_max_pool", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 512]}, {"class_name": "TensorShape", "items": [null]}]}
к
5regularization_losses
6trainable_variables
7	variables
8	keras_api
+И&call_and_return_all_conditional_losses
╣__call__"х
_tf_keras_layerЏ{"class_name": "GlobalAvgPool", "name": "global_avg_pool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_avg_pool", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 512]}, {"class_name": "TensorShape", "items": [null]}]}
к
9regularization_losses
:trainable_variables
;	variables
<	keras_api
+║&call_and_return_all_conditional_losses
╗__call__"х
_tf_keras_layerЏ{"class_name": "GlobalSumPool", "name": "global_sum_pool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_sum_pool", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 512]}, {"class_name": "TensorShape", "items": [null]}]}
C
=0
>1
?2
@3
A4"
trackable_list_wrapper
C
B0
C1
D2
E3
F4"
trackable_list_wrapper
ш

Gkernel
Hbias
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
+╝&call_and_return_all_conditional_losses
й__call__"╬
_tf_keras_layer┤{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
 "
trackable_list_wrapper
ќ
0
M1
N2
O3
P4
Q5
6
 7
&8
,9
R10
S11
T12
U13
V14
W15
X16
Y17
Z18
[19
\20
]21
^22
_23
`24
a25
b26
c27
d28
e29
G30
H31"
trackable_list_wrapper
Т
0
M1
N2
O3
P4
Q5
6
 7
&8
,9
R10
S11
T12
U13
V14
W15
X16
Y17
Z18
[19
\20
]21
^22
_23
`24
a25
b26
c27
d28
e29
f30
g31
h32
i33
j34
k35
l36
m37
n38
o39
G40
H41"
trackable_list_wrapper
╬
player_regularization_losses

qlayers
regularization_losses
rmetrics
trainable_variables
slayer_metrics
tnon_trainable_variables
	variables
ф__call__
Ф_default_save_signature
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
-
Йserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
u0
v1
w2
x3"
trackable_list_wrapper
,:*@2model/ecc_conv/root_kernel
 "
trackable_list_wrapper
J
0
M1
N2
O3
P4
Q5"
trackable_list_wrapper
J
0
M1
N2
O3
P4
Q5"
trackable_list_wrapper
░
ylayer_regularization_losses

zlayers
regularization_losses
{metrics
trainable_variables
|layer_metrics
}non_trainable_variables
	variables
Г__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%@@2model/gcn_conv/kernel
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
│
~layer_regularization_losses

layers
regularization_losses
ђmetrics
trainable_variables
Ђlayer_metrics
ѓnon_trainable_variables
	variables
»__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(	@ђ2model/gcn_conv_1/kernel
 "
trackable_list_wrapper
'
 0"
trackable_list_wrapper
'
 0"
trackable_list_wrapper
х
 Ѓlayer_regularization_losses
ёlayers
!regularization_losses
Ёmetrics
"trainable_variables
єlayer_metrics
Єnon_trainable_variables
#	variables
▒__call__
+░&call_and_return_all_conditional_losses
'░"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)
ђђ2model/gcn_conv_2/kernel
 "
trackable_list_wrapper
'
&0"
trackable_list_wrapper
'
&0"
trackable_list_wrapper
х
 ѕlayer_regularization_losses
Ѕlayers
'regularization_losses
іmetrics
(trainable_variables
Іlayer_metrics
їnon_trainable_variables
)	variables
│__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)
ђђ2model/gcn_conv_3/kernel
 "
trackable_list_wrapper
'
,0"
trackable_list_wrapper
'
,0"
trackable_list_wrapper
х
 Їlayer_regularization_losses
јlayers
-regularization_losses
Јmetrics
.trainable_variables
љlayer_metrics
Љnon_trainable_variables
/	variables
х__call__
+┤&call_and_return_all_conditional_losses
'┤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
 њlayer_regularization_losses
Њlayers
1regularization_losses
ћmetrics
2trainable_variables
Ћlayer_metrics
ќnon_trainable_variables
3	variables
и__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
 Ќlayer_regularization_losses
ўlayers
5regularization_losses
Ўmetrics
6trainable_variables
џlayer_metrics
Џnon_trainable_variables
7	variables
╣__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
 юlayer_regularization_losses
Юlayers
9regularization_losses
ъmetrics
:trainable_variables
Ъlayer_metrics
аnon_trainable_variables
;	variables
╗__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
Щ

Rkernel
Sbias
Аregularization_losses
бtrainable_variables
Б	variables
ц	keras_api
+┐&call_and_return_all_conditional_losses
└__call__"¤
_tf_keras_layerх{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1024, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1536}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1536]}}
§

Tkernel
Ubias
Цregularization_losses
дtrainable_variables
Д	variables
е	keras_api
+┴&call_and_return_all_conditional_losses
┬__call__"м
_tf_keras_layerИ{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}
ч

Vkernel
Wbias
Еregularization_losses
фtrainable_variables
Ф	variables
г	keras_api
+├&call_and_return_all_conditional_losses
─__call__"л
_tf_keras_layerХ{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
ч

Xkernel
Ybias
Гregularization_losses
«trainable_variables
»	variables
░	keras_api
+┼&call_and_return_all_conditional_losses
к__call__"л
_tf_keras_layerХ{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
ч

Zkernel
[bias
▒regularization_losses
▓trainable_variables
│	variables
┤	keras_api
+К&call_and_return_all_conditional_losses
╚__call__"л
_tf_keras_layerХ{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
╣	
	хaxis
	\gamma
]beta
fmoving_mean
gmoving_variance
Хregularization_losses
иtrainable_variables
И	variables
╣	keras_api
+╔&call_and_return_all_conditional_losses
╩__call__"я
_tf_keras_layer─{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}
╗	
	║axis
	^gamma
_beta
hmoving_mean
imoving_variance
╗regularization_losses
╝trainable_variables
й	variables
Й	keras_api
+╦&call_and_return_all_conditional_losses
╠__call__"Я
_tf_keras_layerк{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
╗	
	┐axis
	`gamma
abeta
jmoving_mean
kmoving_variance
└regularization_losses
┴trainable_variables
┬	variables
├	keras_api
+═&call_and_return_all_conditional_losses
╬__call__"Я
_tf_keras_layerк{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
╗	
	─axis
	bgamma
cbeta
lmoving_mean
mmoving_variance
┼regularization_losses
кtrainable_variables
К	variables
╚	keras_api
+¤&call_and_return_all_conditional_losses
л__call__"Я
_tf_keras_layerк{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
╗	
	╔axis
	dgamma
ebeta
nmoving_mean
omoving_variance
╩regularization_losses
╦trainable_variables
╠	variables
═	keras_api
+Л&call_and_return_all_conditional_losses
м__call__"Я
_tf_keras_layerк{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
':%	ђ2model/dense_5/kernel
 :2model/dense_5/bias
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
х
 ╬layer_regularization_losses
¤layers
Iregularization_losses
лmetrics
Jtrainable_variables
Лlayer_metrics
мnon_trainable_variables
K	variables
й__call__
+╝&call_and_return_all_conditional_losses
'╝"call_and_return_conditional_losses"
_generic_user_object
-:+@2model/ecc_conv/FGN_0/kernel
-:+@@2model/ecc_conv/FGN_1/kernel
-:+@@2model/ecc_conv/FGN_2/kernel
0:.	@└2model/ecc_conv/FGN_out/kernel
*:(└2model/ecc_conv/FGN_out/bias
&:$
ђђ2model/dense/kernel
:ђ2model/dense/bias
(:&
ђђ2model/dense_1/kernel
!:ђ2model/dense_1/bias
(:&
ђђ2model/dense_2/kernel
!:ђ2model/dense_2/bias
(:&
ђђ2model/dense_3/kernel
!:ђ2model/dense_3/bias
(:&
ђђ2model/dense_4/kernel
!:ђ2model/dense_4/bias
.:,ђ2model/batch_normalization/gamma
-:+ђ2model/batch_normalization/beta
0:.ђ2!model/batch_normalization_1/gamma
/:-ђ2 model/batch_normalization_1/beta
0:.ђ2!model/batch_normalization_2/gamma
/:-ђ2 model/batch_normalization_2/beta
0:.ђ2!model/batch_normalization_3/gamma
/:-ђ2 model/batch_normalization_3/beta
0:.ђ2!model/batch_normalization_4/gamma
/:-ђ2 model/batch_normalization_4/beta
6:4ђ (2%model/batch_normalization/moving_mean
::8ђ (2)model/batch_normalization/moving_variance
8:6ђ (2'model/batch_normalization_1/moving_mean
<::ђ (2+model/batch_normalization_1/moving_variance
8:6ђ (2'model/batch_normalization_2/moving_mean
<::ђ (2+model/batch_normalization_2/moving_variance
8:6ђ (2'model/batch_normalization_3/moving_mean
<::ђ (2+model/batch_normalization_3/moving_variance
8:6ђ (2'model/batch_normalization_4/moving_mean
<::ђ (2+model/batch_normalization_4/moving_variance
 "
trackable_list_wrapper
«
0
1
2
3
4
5
6
7
=8
>9
?10
@11
A12
B13
C14
D15
E16
F17
18"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
f
f0
g1
h2
i3
j4
k5
l6
m7
n8
o9"
trackable_list_wrapper
Т

Mkernel
Мregularization_losses
нtrainable_variables
Н	variables
о	keras_api
+М&call_and_return_all_conditional_losses
н__call__"┼
_tf_keras_layerФ{"class_name": "Dense", "name": "FGN_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "FGN_0", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}}
У

Nkernel
Оregularization_losses
пtrainable_variables
┘	variables
┌	keras_api
+Н&call_and_return_all_conditional_losses
о__call__"К
_tf_keras_layerГ{"class_name": "Dense", "name": "FGN_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "FGN_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
У

Okernel
█regularization_losses
▄trainable_variables
П	variables
я	keras_api
+О&call_and_return_all_conditional_losses
п__call__"К
_tf_keras_layerГ{"class_name": "Dense", "name": "FGN_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "FGN_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
щ

Pkernel
Qbias
▀regularization_losses
Яtrainable_variables
р	variables
Р	keras_api
+┘&call_and_return_all_conditional_losses
┌__call__"╬
_tf_keras_layer┤{"class_name": "Dense", "name": "FGN_out", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "FGN_out", "trainable": true, "dtype": "float32", "units": 320, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
 "
trackable_list_wrapper
<
u0
v1
w2
x3"
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
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
И
 сlayer_regularization_losses
Сlayers
Аregularization_losses
тmetrics
бtrainable_variables
Тlayer_metrics
уnon_trainable_variables
Б	variables
└__call__
+┐&call_and_return_all_conditional_losses
'┐"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
И
 Уlayer_regularization_losses
жlayers
Цregularization_losses
Жmetrics
дtrainable_variables
вlayer_metrics
Вnon_trainable_variables
Д	variables
┬__call__
+┴&call_and_return_all_conditional_losses
'┴"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
И
 ьlayer_regularization_losses
Ьlayers
Еregularization_losses
№metrics
фtrainable_variables
­layer_metrics
ыnon_trainable_variables
Ф	variables
─__call__
+├&call_and_return_all_conditional_losses
'├"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
И
 Ыlayer_regularization_losses
зlayers
Гregularization_losses
Зmetrics
«trainable_variables
шlayer_metrics
Шnon_trainable_variables
»	variables
к__call__
+┼&call_and_return_all_conditional_losses
'┼"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
И
 эlayer_regularization_losses
Эlayers
▒regularization_losses
щmetrics
▓trainable_variables
Щlayer_metrics
чnon_trainable_variables
│	variables
╚__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
<
\0
]1
f2
g3"
trackable_list_wrapper
И
 Чlayer_regularization_losses
§layers
Хregularization_losses
■metrics
иtrainable_variables
 layer_metrics
ђnon_trainable_variables
И	variables
╩__call__
+╔&call_and_return_all_conditional_losses
'╔"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
<
^0
_1
h2
i3"
trackable_list_wrapper
И
 Ђlayer_regularization_losses
ѓlayers
╗regularization_losses
Ѓmetrics
╝trainable_variables
ёlayer_metrics
Ёnon_trainable_variables
й	variables
╠__call__
+╦&call_and_return_all_conditional_losses
'╦"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
<
`0
a1
j2
k3"
trackable_list_wrapper
И
 єlayer_regularization_losses
Єlayers
└regularization_losses
ѕmetrics
┴trainable_variables
Ѕlayer_metrics
іnon_trainable_variables
┬	variables
╬__call__
+═&call_and_return_all_conditional_losses
'═"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
<
b0
c1
l2
m3"
trackable_list_wrapper
И
 Іlayer_regularization_losses
їlayers
┼regularization_losses
Їmetrics
кtrainable_variables
јlayer_metrics
Јnon_trainable_variables
К	variables
л__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
<
d0
e1
n2
o3"
trackable_list_wrapper
И
 љlayer_regularization_losses
Љlayers
╩regularization_losses
њmetrics
╦trainable_variables
Њlayer_metrics
ћnon_trainable_variables
╠	variables
м__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
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
'
M0"
trackable_list_wrapper
'
M0"
trackable_list_wrapper
И
 Ћlayer_regularization_losses
ќlayers
Мregularization_losses
Ќmetrics
нtrainable_variables
ўlayer_metrics
Ўnon_trainable_variables
Н	variables
н__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
N0"
trackable_list_wrapper
'
N0"
trackable_list_wrapper
И
 џlayer_regularization_losses
Џlayers
Оregularization_losses
юmetrics
пtrainable_variables
Юlayer_metrics
ъnon_trainable_variables
┘	variables
о__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
O0"
trackable_list_wrapper
'
O0"
trackable_list_wrapper
И
 Ъlayer_regularization_losses
аlayers
█regularization_losses
Аmetrics
▄trainable_variables
бlayer_metrics
Бnon_trainable_variables
П	variables
п__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
И
 цlayer_regularization_losses
Цlayers
▀regularization_losses
дmetrics
Яtrainable_variables
Дlayer_metrics
еnon_trainable_variables
р	variables
┌__call__
+┘&call_and_return_all_conditional_losses
'┘"call_and_return_conditional_losses"
_generic_user_object
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
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
n0
o1"
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
Й2╗
@__inference_model_layer_call_and_return_conditional_losses_62646
@__inference_model_layer_call_and_return_conditional_losses_62886┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ѕ2Ё
%__inference_model_layer_call_fn_63072
%__inference_model_layer_call_fn_62979┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
▓2»
 __inference__wrapped_model_60252і
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *zбw
uбr
і         
@њ='б$
Щ                  
ђSparseTensorSpec
і         	
ь2Ж
C__inference_ecc_conv_layer_call_and_return_conditional_losses_63130б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_ecc_conv_layer_call_fn_63151б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_gcn_conv_layer_call_and_return_conditional_losses_63163б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_gcn_conv_layer_call_fn_63173б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_gcn_conv_1_layer_call_and_return_conditional_losses_63185б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_gcn_conv_1_layer_call_fn_63195б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_gcn_conv_2_layer_call_and_return_conditional_losses_63207б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_gcn_conv_2_layer_call_fn_63217б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_gcn_conv_3_layer_call_and_return_conditional_losses_63229б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_gcn_conv_3_layer_call_fn_63239б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
З2ы
J__inference_global_max_pool_layer_call_and_return_conditional_losses_63245б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┘2о
/__inference_global_max_pool_layer_call_fn_63251б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
З2ы
J__inference_global_avg_pool_layer_call_and_return_conditional_losses_63257б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┘2о
/__inference_global_avg_pool_layer_call_fn_63263б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
З2ы
J__inference_global_sum_pool_layer_call_and_return_conditional_losses_63269б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┘2о
/__inference_global_sum_pool_layer_call_fn_63275б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_dense_5_layer_call_and_return_conditional_losses_63285б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_dense_5_layer_call_fn_63294б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№BВ
#__inference_signature_wrapper_62326args_0args_0_1args_0_2args_0_3args_0_4"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ж2у
@__inference_dense_layer_call_and_return_conditional_losses_63304б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
¤2╠
%__inference_dense_layer_call_fn_63313б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_dense_1_layer_call_and_return_conditional_losses_63323б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_dense_1_layer_call_fn_63332б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_dense_2_layer_call_and_return_conditional_losses_63342б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_dense_2_layer_call_fn_63351б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_dense_3_layer_call_and_return_conditional_losses_63361б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_dense_3_layer_call_fn_63370б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_dense_4_layer_call_and_return_conditional_losses_63380б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_dense_4_layer_call_fn_63389б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┌2О
N__inference_batch_normalization_layer_call_and_return_conditional_losses_63445
N__inference_batch_normalization_layer_call_and_return_conditional_losses_63425┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ц2А
3__inference_batch_normalization_layer_call_fn_63458
3__inference_batch_normalization_layer_call_fn_63471┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
я2█
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_63507
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_63527┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
е2Ц
5__inference_batch_normalization_1_layer_call_fn_63553
5__inference_batch_normalization_1_layer_call_fn_63540┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
я2█
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_63609
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_63589┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
е2Ц
5__inference_batch_normalization_2_layer_call_fn_63622
5__inference_batch_normalization_2_layer_call_fn_63635┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
я2█
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_63691
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_63671┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
е2Ц
5__inference_batch_normalization_3_layer_call_fn_63717
5__inference_batch_normalization_3_layer_call_fn_63704┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
я2█
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_63773
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_63753┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
е2Ц
5__inference_batch_normalization_4_layer_call_fn_63799
5__inference_batch_normalization_4_layer_call_fn_63786┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 д
 __inference__wrapped_model_60252Ђ*MNOPQ &,RSfg]\TUhi_^VWjka`XYlmcbZ[noedGHЮбЎ
ЉбЇ
ібє
"і
args_0/0         
@њ='б$
Щ                  
ђSparseTensorSpec
і
args_0/2         	
ф "3ф0
.
output_1"і
output_1         И
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_63507dhi_^4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ И
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_63527dhi_^4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ љ
5__inference_batch_normalization_1_layer_call_fn_63540Whi_^4б1
*б'
!і
inputs         ђ
p
ф "і         ђљ
5__inference_batch_normalization_1_layer_call_fn_63553Whi_^4б1
*б'
!і
inputs         ђ
p 
ф "і         ђИ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_63589djka`4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ И
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_63609djka`4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ љ
5__inference_batch_normalization_2_layer_call_fn_63622Wjka`4б1
*б'
!і
inputs         ђ
p
ф "і         ђљ
5__inference_batch_normalization_2_layer_call_fn_63635Wjka`4б1
*б'
!і
inputs         ђ
p 
ф "і         ђИ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_63671dlmcb4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ И
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_63691dlmcb4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ љ
5__inference_batch_normalization_3_layer_call_fn_63704Wlmcb4б1
*б'
!і
inputs         ђ
p
ф "і         ђљ
5__inference_batch_normalization_3_layer_call_fn_63717Wlmcb4б1
*б'
!і
inputs         ђ
p 
ф "і         ђИ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_63753dnoed4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ И
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_63773dnoed4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ љ
5__inference_batch_normalization_4_layer_call_fn_63786Wnoed4б1
*б'
!і
inputs         ђ
p
ф "і         ђљ
5__inference_batch_normalization_4_layer_call_fn_63799Wnoed4б1
*б'
!і
inputs         ђ
p 
ф "і         ђХ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_63425dfg]\4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ Х
N__inference_batch_normalization_layer_call_and_return_conditional_losses_63445dfg]\4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ ј
3__inference_batch_normalization_layer_call_fn_63458Wfg]\4б1
*б'
!і
inputs         ђ
p
ф "і         ђј
3__inference_batch_normalization_layer_call_fn_63471Wfg]\4б1
*б'
!і
inputs         ђ
p 
ф "і         ђц
B__inference_dense_1_layer_call_and_return_conditional_losses_63323^TU0б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ |
'__inference_dense_1_layer_call_fn_63332QTU0б-
&б#
!і
inputs         ђ
ф "і         ђц
B__inference_dense_2_layer_call_and_return_conditional_losses_63342^VW0б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ |
'__inference_dense_2_layer_call_fn_63351QVW0б-
&б#
!і
inputs         ђ
ф "і         ђц
B__inference_dense_3_layer_call_and_return_conditional_losses_63361^XY0б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ |
'__inference_dense_3_layer_call_fn_63370QXY0б-
&б#
!і
inputs         ђ
ф "і         ђц
B__inference_dense_4_layer_call_and_return_conditional_losses_63380^Z[0б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ |
'__inference_dense_4_layer_call_fn_63389QZ[0б-
&б#
!і
inputs         ђ
ф "і         ђБ
B__inference_dense_5_layer_call_and_return_conditional_losses_63285]GH0б-
&б#
!і
inputs         ђ
ф "%б"
і
0         
џ {
'__inference_dense_5_layer_call_fn_63294PGH0б-
&б#
!і
inputs         ђ
ф "і         б
@__inference_dense_layer_call_and_return_conditional_losses_63304^RS0б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ z
%__inference_dense_layer_call_fn_63313QRS0б-
&б#
!і
inputs         ђ
ф "і         ђЏ
C__inference_ecc_conv_layer_call_and_return_conditional_losses_63130МMNOPQАбЮ
ЋбЉ
јџі
"і
inputs/0         
@њ='б$
Щ                  
ђSparseTensorSpec
"і
inputs/2         
ф "%б"
і
0         @
џ з
(__inference_ecc_conv_layer_call_fn_63151кMNOPQАбЮ
ЋбЉ
јџі
"і
inputs/0         
@њ='б$
Щ                  
ђSparseTensorSpec
"і
inputs/2         
ф "і         @№
E__inference_gcn_conv_1_layer_call_and_return_conditional_losses_63185Ц xбu
nбk
iџf
"і
inputs/0         @
@њ='б$
Щ                  
ђSparseTensorSpec
ф "&б#
і
0         ђ
џ К
*__inference_gcn_conv_1_layer_call_fn_63195ў xбu
nбk
iџf
"і
inputs/0         @
@њ='б$
Щ                  
ђSparseTensorSpec
ф "і         ђ­
E__inference_gcn_conv_2_layer_call_and_return_conditional_losses_63207д&yбv
oбl
jџg
#і 
inputs/0         ђ
@њ='б$
Щ                  
ђSparseTensorSpec
ф "&б#
і
0         ђ
џ ╚
*__inference_gcn_conv_2_layer_call_fn_63217Ў&yбv
oбl
jџg
#і 
inputs/0         ђ
@њ='б$
Щ                  
ђSparseTensorSpec
ф "і         ђ­
E__inference_gcn_conv_3_layer_call_and_return_conditional_losses_63229д,yбv
oбl
jџg
#і 
inputs/0         ђ
@њ='б$
Щ                  
ђSparseTensorSpec
ф "&б#
і
0         ђ
џ ╚
*__inference_gcn_conv_3_layer_call_fn_63239Ў,yбv
oбl
jџg
#і 
inputs/0         ђ
@њ='б$
Щ                  
ђSparseTensorSpec
ф "і         ђВ
C__inference_gcn_conv_layer_call_and_return_conditional_losses_63163цxбu
nбk
iџf
"і
inputs/0         @
@њ='б$
Щ                  
ђSparseTensorSpec
ф "%б"
і
0         @
џ ─
(__inference_gcn_conv_layer_call_fn_63173Ќxбu
nбk
iџf
"і
inputs/0         @
@њ='б$
Щ                  
ђSparseTensorSpec
ф "і         @л
J__inference_global_avg_pool_layer_call_and_return_conditional_losses_63257ЂWбT
MбJ
HџE
#і 
inputs/0         ђ
і
inputs/1         	
ф "&б#
і
0         ђ
џ Д
/__inference_global_avg_pool_layer_call_fn_63263tWбT
MбJ
HџE
#і 
inputs/0         ђ
і
inputs/1         	
ф "і         ђл
J__inference_global_max_pool_layer_call_and_return_conditional_losses_63245ЂWбT
MбJ
HџE
#і 
inputs/0         ђ
і
inputs/1         	
ф "&б#
і
0         ђ
џ Д
/__inference_global_max_pool_layer_call_fn_63251tWбT
MбJ
HџE
#і 
inputs/0         ђ
і
inputs/1         	
ф "і         ђл
J__inference_global_sum_pool_layer_call_and_return_conditional_losses_63269ЂWбT
MбJ
HџE
#і 
inputs/0         ђ
і
inputs/1         	
ф "&б#
і
0         ђ
џ Д
/__inference_global_sum_pool_layer_call_fn_63275tWбT
MбJ
HџE
#і 
inputs/0         ђ
і
inputs/1         	
ф "і         ђ╝
@__inference_model_layer_call_and_return_conditional_losses_62646э*MNOPQ &,RSfg]\TUhi_^VWjka`XYlmcbZ[noedGHАбЮ
ЋбЉ
ібє
"і
inputs/0         
@њ='б$
Щ                  
ђSparseTensorSpec
і
inputs/2         	
p
ф "%б"
і
0         
џ ╝
@__inference_model_layer_call_and_return_conditional_losses_62886э*MNOPQ &,RSfg]\TUhi_^VWjka`XYlmcbZ[noedGHАбЮ
ЋбЉ
ібє
"і
inputs/0         
@њ='б$
Щ                  
ђSparseTensorSpec
і
inputs/2         	
p 
ф "%б"
і
0         
џ ћ
%__inference_model_layer_call_fn_62979Ж*MNOPQ &,RSfg]\TUhi_^VWjka`XYlmcbZ[noedGHАбЮ
ЋбЉ
ібє
"і
inputs/0         
@њ='б$
Щ                  
ђSparseTensorSpec
і
inputs/2         	
p
ф "і         ћ
%__inference_model_layer_call_fn_63072Ж*MNOPQ &,RSfg]\TUhi_^VWjka`XYlmcbZ[noedGHАбЮ
ЋбЉ
ібє
"і
inputs/0         
@њ='б$
Щ                  
ђSparseTensorSpec
і
inputs/2         	
p 
ф "і         з
#__inference_signature_wrapper_62326╦*MNOPQ &,RSfg]\TUhi_^VWjka`XYlmcbZ[noedGHубс
б 
█фО
*
args_0 і
args_0         
.
args_0_1"і
args_0_1         	
*
args_0_2і
args_0_2         
!
args_0_3і
args_0_3	
*
args_0_4і
args_0_4         	"3ф0
.
output_1"і
output_1         