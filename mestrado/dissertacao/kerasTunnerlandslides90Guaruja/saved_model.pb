��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
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
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
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
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718��
\
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namemean
U
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
: *
dtype0
d
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
variance
]
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0	
`
mean_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namemean_1
Y
mean_1/Read/ReadVariableOpReadVariableOpmean_1*
_output_shapes
: *
dtype0
h

variance_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
variance_1
a
variance_1/Read/ReadVariableOpReadVariableOp
variance_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0	
`
mean_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namemean_2
Y
mean_2/Read/ReadVariableOpReadVariableOpmean_2*
_output_shapes
: *
dtype0
h

variance_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
variance_2
a
variance_2/Read/ReadVariableOpReadVariableOp
variance_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0	
`
mean_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namemean_3
Y
mean_3/Read/ReadVariableOpReadVariableOpmean_3*
_output_shapes
: *
dtype0
h

variance_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
variance_3
a
variance_3/Read/ReadVariableOpReadVariableOp
variance_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0	
`
mean_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namemean_4
Y
mean_4/Read/ReadVariableOpReadVariableOpmean_4*
_output_shapes
: *
dtype0
h

variance_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
variance_4
a
variance_4/Read/ReadVariableOpReadVariableOp
variance_4*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0	
`
mean_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namemean_5
Y
mean_5/Read/ReadVariableOpReadVariableOpmean_5*
_output_shapes
: *
dtype0
h

variance_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
variance_5
a
variance_5/Read/ReadVariableOpReadVariableOp
variance_5*
_output_shapes
: *
dtype0
b
count_5VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_5
[
count_5/Read/ReadVariableOpReadVariableOpcount_5*
_output_shapes
: *
dtype0	
`
mean_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namemean_6
Y
mean_6/Read/ReadVariableOpReadVariableOpmean_6*
_output_shapes
: *
dtype0
h

variance_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
variance_6
a
variance_6/Read/ReadVariableOpReadVariableOp
variance_6*
_output_shapes
: *
dtype0
b
count_6VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_6
[
count_6/Read/ReadVariableOpReadVariableOpcount_6*
_output_shapes
: *
dtype0	
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_7
[
count_7/Read/ReadVariableOpReadVariableOpcount_7*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_8
[
count_8/Read/ReadVariableOpReadVariableOpcount_8*
_output_shapes
: *
dtype0
�
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�=
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�<
value�<B�< B�<
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-0
layer-7
	layer_with_weights-1
	layer-8

layer_with_weights-2

layer-9
layer_with_weights-3
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer-16
layer_with_weights-8
layer-17
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
 
 
 
 
 
 
�

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean
variance
	count
 	keras_api
�
!
_keep_axis
"_reduce_axis
#_reduce_axis_mask
$_broadcast_shape
%mean
&variance
	'count
(	keras_api
�
)
_keep_axis
*_reduce_axis
+_reduce_axis_mask
,_broadcast_shape
-mean
.variance
	/count
0	keras_api
�
1
_keep_axis
2_reduce_axis
3_reduce_axis_mask
4_broadcast_shape
5mean
6variance
	7count
8	keras_api
�
9
_keep_axis
:_reduce_axis
;_reduce_axis_mask
<_broadcast_shape
=mean
>variance
	?count
@	keras_api
�
A
_keep_axis
B_reduce_axis
C_reduce_axis_mask
D_broadcast_shape
Emean
Fvariance
	Gcount
H	keras_api
�
I
_keep_axis
J_reduce_axis
K_reduce_axis_mask
L_broadcast_shape
Mmean
Nvariance
	Ocount
P	keras_api
R
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
h

Ukernel
Vbias
Wtrainable_variables
Xregularization_losses
Y	variables
Z	keras_api
R
[trainable_variables
\regularization_losses
]	variables
^	keras_api
h

_kernel
`bias
atrainable_variables
bregularization_losses
c	variables
d	keras_api
�
eiter

fbeta_1

gbeta_2
	hdecay
ilearning_rateUm�Vm�_m�`m�Uv�Vv�_v�`v�

U0
V1
_2
`3
 
�
0
1
2
%3
&4
'5
-6
.7
/8
59
610
711
=12
>13
?14
E15
F16
G17
M18
N19
O20
U21
V22
_23
`24
�
trainable_variables
jlayer_regularization_losses
knon_trainable_variables

llayers
mmetrics
regularization_losses
nlayer_metrics
	variables
 
 
 
 
 
NL
VARIABLE_VALUEmean4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEvariance8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEcount5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_14layer_with_weights-1/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_18layer_with_weights-1/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_15layer_with_weights-1/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_24layer_with_weights-2/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_28layer_with_weights-2/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_25layer_with_weights-2/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_34layer_with_weights-3/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_38layer_with_weights-3/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_35layer_with_weights-3/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_44layer_with_weights-4/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_48layer_with_weights-4/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_45layer_with_weights-4/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_54layer_with_weights-5/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_58layer_with_weights-5/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_55layer_with_weights-5/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_64layer_with_weights-6/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_68layer_with_weights-6/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_65layer_with_weights-6/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
�
Qtrainable_variables
olayer_regularization_losses
pnon_trainable_variables

qlayers
rmetrics
Rregularization_losses
slayer_metrics
S	variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

U0
V1
 

U0
V1
�
Wtrainable_variables
tlayer_regularization_losses
unon_trainable_variables

vlayers
wmetrics
Xregularization_losses
xlayer_metrics
Y	variables
 
 
 
�
[trainable_variables
ylayer_regularization_losses
znon_trainable_variables

{layers
|metrics
\regularization_losses
}layer_metrics
]	variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

_0
`1
 

_0
`1
�
atrainable_variables
~layer_regularization_losses
non_trainable_variables
�layers
�metrics
bregularization_losses
�layer_metrics
c	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
�
0
1
2
%3
&4
'5
-6
.7
/8
59
610
711
=12
>13
?14
E15
F16
G17
M18
N19
O20
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17

�0
�1
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
8

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_74keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_84keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
y
serving_default_aspectPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
w
serving_default_curvPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
|
serving_default_elevationPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
w
serving_default_litoPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
x
serving_default_slopePlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
v
serving_default_twiPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
{
serving_default_uso_soloPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_aspectserving_default_curvserving_default_elevationserving_default_litoserving_default_slopeserving_default_twiserving_default_uso_solomeanvariancemean_1
variance_1mean_2
variance_2mean_3
variance_3mean_4
variance_4mean_5
variance_5mean_6
variance_6dense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_319981
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOpmean_1/Read/ReadVariableOpvariance_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpmean_2/Read/ReadVariableOpvariance_2/Read/ReadVariableOpcount_2/Read/ReadVariableOpmean_3/Read/ReadVariableOpvariance_3/Read/ReadVariableOpcount_3/Read/ReadVariableOpmean_4/Read/ReadVariableOpvariance_4/Read/ReadVariableOpcount_4/Read/ReadVariableOpmean_5/Read/ReadVariableOpvariance_5/Read/ReadVariableOpcount_5/Read/ReadVariableOpmean_6/Read/ReadVariableOpvariance_6/Read/ReadVariableOpcount_6/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount_7/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_8/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOpConst*7
Tin0
.2,								*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_320903
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecountmean_1
variance_1count_1mean_2
variance_2count_2mean_3
variance_3count_3mean_4
variance_4count_4mean_5
variance_5count_5mean_6
variance_6count_6dense_1/kerneldense_1/biasdense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount_7total_1count_8Adam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/v*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_321039��
�
�
&__inference_model_layer_call_fn_320075
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: 

unknown_13:

unknown_14:

unknown_15:

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_3196142
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/6
��
�
A__inference_model_layer_call_and_return_conditional_losses_320192
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_69
/normalization_1_reshape_readvariableop_resource: ;
1normalization_1_reshape_1_readvariableop_resource: 9
/normalization_2_reshape_readvariableop_resource: ;
1normalization_2_reshape_1_readvariableop_resource: 9
/normalization_3_reshape_readvariableop_resource: ;
1normalization_3_reshape_1_readvariableop_resource: 9
/normalization_4_reshape_readvariableop_resource: ;
1normalization_4_reshape_1_readvariableop_resource: 9
/normalization_5_reshape_readvariableop_resource: ;
1normalization_5_reshape_1_readvariableop_resource: 9
/normalization_6_reshape_readvariableop_resource: ;
1normalization_6_reshape_1_readvariableop_resource: 9
/normalization_7_reshape_readvariableop_resource: ;
1normalization_7_reshape_1_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identity��dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�&normalization_1/Reshape/ReadVariableOp�(normalization_1/Reshape_1/ReadVariableOp�&normalization_2/Reshape/ReadVariableOp�(normalization_2/Reshape_1/ReadVariableOp�&normalization_3/Reshape/ReadVariableOp�(normalization_3/Reshape_1/ReadVariableOp�&normalization_4/Reshape/ReadVariableOp�(normalization_4/Reshape_1/ReadVariableOp�&normalization_5/Reshape/ReadVariableOp�(normalization_5/Reshape_1/ReadVariableOp�&normalization_6/Reshape/ReadVariableOp�(normalization_6/Reshape_1/ReadVariableOp�&normalization_7/Reshape/ReadVariableOp�(normalization_7/Reshape_1/ReadVariableOp�
&normalization_1/Reshape/ReadVariableOpReadVariableOp/normalization_1_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_1/Reshape/ReadVariableOp�
normalization_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_1/Reshape/shape�
normalization_1/ReshapeReshape.normalization_1/Reshape/ReadVariableOp:value:0&normalization_1/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_1/Reshape�
(normalization_1/Reshape_1/ReadVariableOpReadVariableOp1normalization_1_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_1/Reshape_1/ReadVariableOp�
normalization_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_1/Reshape_1/shape�
normalization_1/Reshape_1Reshape0normalization_1/Reshape_1/ReadVariableOp:value:0(normalization_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_1/Reshape_1�
normalization_1/subSubinputs_0 normalization_1/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_1/sub�
normalization_1/SqrtSqrt"normalization_1/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_1/Sqrt{
normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_1/Maximum/y�
normalization_1/MaximumMaximumnormalization_1/Sqrt:y:0"normalization_1/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_1/Maximum�
normalization_1/truedivRealDivnormalization_1/sub:z:0normalization_1/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_1/truediv�
&normalization_2/Reshape/ReadVariableOpReadVariableOp/normalization_2_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_2/Reshape/ReadVariableOp�
normalization_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_2/Reshape/shape�
normalization_2/ReshapeReshape.normalization_2/Reshape/ReadVariableOp:value:0&normalization_2/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_2/Reshape�
(normalization_2/Reshape_1/ReadVariableOpReadVariableOp1normalization_2_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_2/Reshape_1/ReadVariableOp�
normalization_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_2/Reshape_1/shape�
normalization_2/Reshape_1Reshape0normalization_2/Reshape_1/ReadVariableOp:value:0(normalization_2/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_2/Reshape_1�
normalization_2/subSubinputs_1 normalization_2/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_2/sub�
normalization_2/SqrtSqrt"normalization_2/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_2/Sqrt{
normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_2/Maximum/y�
normalization_2/MaximumMaximumnormalization_2/Sqrt:y:0"normalization_2/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_2/Maximum�
normalization_2/truedivRealDivnormalization_2/sub:z:0normalization_2/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_2/truediv�
&normalization_3/Reshape/ReadVariableOpReadVariableOp/normalization_3_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_3/Reshape/ReadVariableOp�
normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_3/Reshape/shape�
normalization_3/ReshapeReshape.normalization_3/Reshape/ReadVariableOp:value:0&normalization_3/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_3/Reshape�
(normalization_3/Reshape_1/ReadVariableOpReadVariableOp1normalization_3_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_3/Reshape_1/ReadVariableOp�
normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_3/Reshape_1/shape�
normalization_3/Reshape_1Reshape0normalization_3/Reshape_1/ReadVariableOp:value:0(normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_3/Reshape_1�
normalization_3/subSubinputs_2 normalization_3/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_3/sub�
normalization_3/SqrtSqrt"normalization_3/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_3/Maximum/y�
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_3/Maximum�
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_3/truediv�
&normalization_4/Reshape/ReadVariableOpReadVariableOp/normalization_4_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_4/Reshape/ReadVariableOp�
normalization_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_4/Reshape/shape�
normalization_4/ReshapeReshape.normalization_4/Reshape/ReadVariableOp:value:0&normalization_4/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_4/Reshape�
(normalization_4/Reshape_1/ReadVariableOpReadVariableOp1normalization_4_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_4/Reshape_1/ReadVariableOp�
normalization_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_4/Reshape_1/shape�
normalization_4/Reshape_1Reshape0normalization_4/Reshape_1/ReadVariableOp:value:0(normalization_4/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_4/Reshape_1�
normalization_4/subSubinputs_3 normalization_4/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_4/sub�
normalization_4/SqrtSqrt"normalization_4/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_4/Sqrt{
normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_4/Maximum/y�
normalization_4/MaximumMaximumnormalization_4/Sqrt:y:0"normalization_4/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_4/Maximum�
normalization_4/truedivRealDivnormalization_4/sub:z:0normalization_4/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_4/truediv�
&normalization_5/Reshape/ReadVariableOpReadVariableOp/normalization_5_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_5/Reshape/ReadVariableOp�
normalization_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_5/Reshape/shape�
normalization_5/ReshapeReshape.normalization_5/Reshape/ReadVariableOp:value:0&normalization_5/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_5/Reshape�
(normalization_5/Reshape_1/ReadVariableOpReadVariableOp1normalization_5_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_5/Reshape_1/ReadVariableOp�
normalization_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_5/Reshape_1/shape�
normalization_5/Reshape_1Reshape0normalization_5/Reshape_1/ReadVariableOp:value:0(normalization_5/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_5/Reshape_1�
normalization_5/subSubinputs_4 normalization_5/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_5/sub�
normalization_5/SqrtSqrt"normalization_5/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_5/Sqrt{
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_5/Maximum/y�
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_5/Maximum�
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_5/truediv�
&normalization_6/Reshape/ReadVariableOpReadVariableOp/normalization_6_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_6/Reshape/ReadVariableOp�
normalization_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_6/Reshape/shape�
normalization_6/ReshapeReshape.normalization_6/Reshape/ReadVariableOp:value:0&normalization_6/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_6/Reshape�
(normalization_6/Reshape_1/ReadVariableOpReadVariableOp1normalization_6_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_6/Reshape_1/ReadVariableOp�
normalization_6/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_6/Reshape_1/shape�
normalization_6/Reshape_1Reshape0normalization_6/Reshape_1/ReadVariableOp:value:0(normalization_6/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_6/Reshape_1�
normalization_6/subSubinputs_5 normalization_6/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_6/sub�
normalization_6/SqrtSqrt"normalization_6/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_6/Sqrt{
normalization_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_6/Maximum/y�
normalization_6/MaximumMaximumnormalization_6/Sqrt:y:0"normalization_6/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_6/Maximum�
normalization_6/truedivRealDivnormalization_6/sub:z:0normalization_6/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_6/truediv�
&normalization_7/Reshape/ReadVariableOpReadVariableOp/normalization_7_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_7/Reshape/ReadVariableOp�
normalization_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_7/Reshape/shape�
normalization_7/ReshapeReshape.normalization_7/Reshape/ReadVariableOp:value:0&normalization_7/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_7/Reshape�
(normalization_7/Reshape_1/ReadVariableOpReadVariableOp1normalization_7_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_7/Reshape_1/ReadVariableOp�
normalization_7/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_7/Reshape_1/shape�
normalization_7/Reshape_1Reshape0normalization_7/Reshape_1/ReadVariableOp:value:0(normalization_7/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_7/Reshape_1�
normalization_7/subSubinputs_6 normalization_7/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_7/sub�
normalization_7/SqrtSqrt"normalization_7/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_7/Sqrt{
normalization_7/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_7/Maximum/y�
normalization_7/MaximumMaximumnormalization_7/Sqrt:y:0"normalization_7/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_7/Maximum�
normalization_7/truedivRealDivnormalization_7/sub:z:0normalization_7/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_7/truedivt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis�
concatenate/concatConcatV2normalization_1/truediv:z:0normalization_2/truediv:z:0normalization_3/truediv:z:0normalization_4/truediv:z:0normalization_5/truediv:z:0normalization_6/truediv:z:0normalization_7/truediv:z:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatenate/concat�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulconcatenate/concat:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/BiasAddp
dense_1/TanhTanhdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_1/Tanhx
dropout_1/IdentityIdentitydense_1/Tanh:y:0*
T0*'
_output_shapes
:���������2
dropout_1/Identity�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMuldropout_1/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_2/BiasAdd�
IdentityIdentitydense_2/BiasAdd:output:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp'^normalization_1/Reshape/ReadVariableOp)^normalization_1/Reshape_1/ReadVariableOp'^normalization_2/Reshape/ReadVariableOp)^normalization_2/Reshape_1/ReadVariableOp'^normalization_3/Reshape/ReadVariableOp)^normalization_3/Reshape_1/ReadVariableOp'^normalization_4/Reshape/ReadVariableOp)^normalization_4/Reshape_1/ReadVariableOp'^normalization_5/Reshape/ReadVariableOp)^normalization_5/Reshape_1/ReadVariableOp'^normalization_6/Reshape/ReadVariableOp)^normalization_6/Reshape_1/ReadVariableOp'^normalization_7/Reshape/ReadVariableOp)^normalization_7/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������: : : : : : : : : : : : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2P
&normalization_1/Reshape/ReadVariableOp&normalization_1/Reshape/ReadVariableOp2T
(normalization_1/Reshape_1/ReadVariableOp(normalization_1/Reshape_1/ReadVariableOp2P
&normalization_2/Reshape/ReadVariableOp&normalization_2/Reshape/ReadVariableOp2T
(normalization_2/Reshape_1/ReadVariableOp(normalization_2/Reshape_1/ReadVariableOp2P
&normalization_3/Reshape/ReadVariableOp&normalization_3/Reshape/ReadVariableOp2T
(normalization_3/Reshape_1/ReadVariableOp(normalization_3/Reshape_1/ReadVariableOp2P
&normalization_4/Reshape/ReadVariableOp&normalization_4/Reshape/ReadVariableOp2T
(normalization_4/Reshape_1/ReadVariableOp(normalization_4/Reshape_1/ReadVariableOp2P
&normalization_5/Reshape/ReadVariableOp&normalization_5/Reshape/ReadVariableOp2T
(normalization_5/Reshape_1/ReadVariableOp(normalization_5/Reshape_1/ReadVariableOp2P
&normalization_6/Reshape/ReadVariableOp&normalization_6/Reshape/ReadVariableOp2T
(normalization_6/Reshape_1/ReadVariableOp(normalization_6/Reshape_1/ReadVariableOp2P
&normalization_7/Reshape/ReadVariableOp&normalization_7/Reshape/ReadVariableOp2T
(normalization_7/Reshape_1/ReadVariableOp(normalization_7/Reshape_1/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/6
�N
�
__inference__traced_save_320903
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	%
!savev2_mean_1_read_readvariableop)
%savev2_variance_1_read_readvariableop&
"savev2_count_1_read_readvariableop	%
!savev2_mean_2_read_readvariableop)
%savev2_variance_2_read_readvariableop&
"savev2_count_2_read_readvariableop	%
!savev2_mean_3_read_readvariableop)
%savev2_variance_3_read_readvariableop&
"savev2_count_3_read_readvariableop	%
!savev2_mean_4_read_readvariableop)
%savev2_variance_4_read_readvariableop&
"savev2_count_4_read_readvariableop	%
!savev2_mean_5_read_readvariableop)
%savev2_variance_5_read_readvariableop&
"savev2_count_5_read_readvariableop	%
!savev2_mean_6_read_readvariableop)
%savev2_variance_6_read_readvariableop&
"savev2_count_6_read_readvariableop	-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_7_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_8_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop
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
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*�
value�B�+B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-1/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-2/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-3/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-4/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-5/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-6/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop!savev2_mean_1_read_readvariableop%savev2_variance_1_read_readvariableop"savev2_count_1_read_readvariableop!savev2_mean_2_read_readvariableop%savev2_variance_2_read_readvariableop"savev2_count_2_read_readvariableop!savev2_mean_3_read_readvariableop%savev2_variance_3_read_readvariableop"savev2_count_3_read_readvariableop!savev2_mean_4_read_readvariableop%savev2_variance_4_read_readvariableop"savev2_count_4_read_readvariableop!savev2_mean_5_read_readvariableop%savev2_variance_5_read_readvariableop"savev2_count_5_read_readvariableop!savev2_mean_6_read_readvariableop%savev2_variance_6_read_readvariableop"savev2_count_6_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop"savev2_count_7_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_8_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *9
dtypes/
-2+								2
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
�: : : : : : : : : : : : : : : : : : : : : : ::::: : : : : : : : : ::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :$# 

_output_shapes

:: $

_output_shapes
::$% 

_output_shapes

:: &

_output_shapes
::$' 

_output_shapes

:: (

_output_shapes
::$) 

_output_shapes

:: *

_output_shapes
::+

_output_shapes
: 
�
c
*__inference_dropout_1_layer_call_fn_320712

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_3194212
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
"__inference__traced_restore_321039
file_prefix
assignvariableop_mean: %
assignvariableop_1_variance: "
assignvariableop_2_count:	 #
assignvariableop_3_mean_1: '
assignvariableop_4_variance_1: $
assignvariableop_5_count_1:	 #
assignvariableop_6_mean_2: '
assignvariableop_7_variance_2: $
assignvariableop_8_count_2:	 #
assignvariableop_9_mean_3: (
assignvariableop_10_variance_3: %
assignvariableop_11_count_3:	 $
assignvariableop_12_mean_4: (
assignvariableop_13_variance_4: %
assignvariableop_14_count_4:	 $
assignvariableop_15_mean_5: (
assignvariableop_16_variance_5: %
assignvariableop_17_count_5:	 $
assignvariableop_18_mean_6: (
assignvariableop_19_variance_6: %
assignvariableop_20_count_6:	 4
"assignvariableop_21_dense_1_kernel:.
 assignvariableop_22_dense_1_bias:4
"assignvariableop_23_dense_2_kernel:.
 assignvariableop_24_dense_2_bias:'
assignvariableop_25_adam_iter:	 )
assignvariableop_26_adam_beta_1: )
assignvariableop_27_adam_beta_2: (
assignvariableop_28_adam_decay: 0
&assignvariableop_29_adam_learning_rate: #
assignvariableop_30_total: %
assignvariableop_31_count_7: %
assignvariableop_32_total_1: %
assignvariableop_33_count_8: ;
)assignvariableop_34_adam_dense_1_kernel_m:5
'assignvariableop_35_adam_dense_1_bias_m:;
)assignvariableop_36_adam_dense_2_kernel_m:5
'assignvariableop_37_adam_dense_2_bias_m:;
)assignvariableop_38_adam_dense_1_kernel_v:5
'assignvariableop_39_adam_dense_1_bias_v:;
)assignvariableop_40_adam_dense_2_kernel_v:5
'assignvariableop_41_adam_dense_2_bias_v:
identity_43��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*�
value�B�+B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-1/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-2/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-3/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-4/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-5/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-6/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+								2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_meanIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_varianceIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_countIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_mean_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variance_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_count_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_mean_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_variance_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_count_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_mean_3Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_variance_3Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_3Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_mean_4Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_variance_4Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_4Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_mean_5Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpassignvariableop_16_variance_5Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_5Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpassignvariableop_18_mean_6Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpassignvariableop_19_variance_6Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_6Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp"assignvariableop_21_dense_1_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp assignvariableop_22_dense_1_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp"assignvariableop_23_dense_2_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp assignvariableop_24_dense_2_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_iterIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_beta_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_beta_2Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOpassignvariableop_28_adam_decayIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp&assignvariableop_29_adam_learning_rateIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOpassignvariableop_30_totalIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOpassignvariableop_31_count_7Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOpassignvariableop_32_total_1Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOpassignvariableop_33_count_8Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_dense_1_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_dense_1_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_dense_2_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_dense_2_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_dense_1_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp'assignvariableop_39_adam_dense_1_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_dense_2_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp'assignvariableop_41_adam_dense_2_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_419
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_42Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_42�
Identity_43IdentityIdentity_42:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_43"#
identity_43Identity_43:output:0*i
_input_shapesX
V: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
�

�
G__inference_concatenate_layer_call_and_return_conditional_losses_320682
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/6
�
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_320717

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
,__inference_concatenate_layer_call_fn_320670
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_3193092
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/6
�
�
&__inference_model_layer_call_fn_320028
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: 

unknown_13:

unknown_14:

unknown_15:

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_3193522
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/6
�
�
A__inference_model_layer_call_and_return_conditional_losses_319352

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_69
/normalization_1_reshape_readvariableop_resource: ;
1normalization_1_reshape_1_readvariableop_resource: 9
/normalization_2_reshape_readvariableop_resource: ;
1normalization_2_reshape_1_readvariableop_resource: 9
/normalization_3_reshape_readvariableop_resource: ;
1normalization_3_reshape_1_readvariableop_resource: 9
/normalization_4_reshape_readvariableop_resource: ;
1normalization_4_reshape_1_readvariableop_resource: 9
/normalization_5_reshape_readvariableop_resource: ;
1normalization_5_reshape_1_readvariableop_resource: 9
/normalization_6_reshape_readvariableop_resource: ;
1normalization_6_reshape_1_readvariableop_resource: 9
/normalization_7_reshape_readvariableop_resource: ;
1normalization_7_reshape_1_readvariableop_resource:  
dense_1_319323:
dense_1_319325: 
dense_2_319346:
dense_2_319348:
identity��dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�&normalization_1/Reshape/ReadVariableOp�(normalization_1/Reshape_1/ReadVariableOp�&normalization_2/Reshape/ReadVariableOp�(normalization_2/Reshape_1/ReadVariableOp�&normalization_3/Reshape/ReadVariableOp�(normalization_3/Reshape_1/ReadVariableOp�&normalization_4/Reshape/ReadVariableOp�(normalization_4/Reshape_1/ReadVariableOp�&normalization_5/Reshape/ReadVariableOp�(normalization_5/Reshape_1/ReadVariableOp�&normalization_6/Reshape/ReadVariableOp�(normalization_6/Reshape_1/ReadVariableOp�&normalization_7/Reshape/ReadVariableOp�(normalization_7/Reshape_1/ReadVariableOp�
&normalization_1/Reshape/ReadVariableOpReadVariableOp/normalization_1_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_1/Reshape/ReadVariableOp�
normalization_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_1/Reshape/shape�
normalization_1/ReshapeReshape.normalization_1/Reshape/ReadVariableOp:value:0&normalization_1/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_1/Reshape�
(normalization_1/Reshape_1/ReadVariableOpReadVariableOp1normalization_1_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_1/Reshape_1/ReadVariableOp�
normalization_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_1/Reshape_1/shape�
normalization_1/Reshape_1Reshape0normalization_1/Reshape_1/ReadVariableOp:value:0(normalization_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_1/Reshape_1�
normalization_1/subSubinputs normalization_1/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_1/sub�
normalization_1/SqrtSqrt"normalization_1/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_1/Sqrt{
normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_1/Maximum/y�
normalization_1/MaximumMaximumnormalization_1/Sqrt:y:0"normalization_1/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_1/Maximum�
normalization_1/truedivRealDivnormalization_1/sub:z:0normalization_1/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_1/truediv�
&normalization_2/Reshape/ReadVariableOpReadVariableOp/normalization_2_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_2/Reshape/ReadVariableOp�
normalization_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_2/Reshape/shape�
normalization_2/ReshapeReshape.normalization_2/Reshape/ReadVariableOp:value:0&normalization_2/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_2/Reshape�
(normalization_2/Reshape_1/ReadVariableOpReadVariableOp1normalization_2_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_2/Reshape_1/ReadVariableOp�
normalization_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_2/Reshape_1/shape�
normalization_2/Reshape_1Reshape0normalization_2/Reshape_1/ReadVariableOp:value:0(normalization_2/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_2/Reshape_1�
normalization_2/subSubinputs_1 normalization_2/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_2/sub�
normalization_2/SqrtSqrt"normalization_2/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_2/Sqrt{
normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_2/Maximum/y�
normalization_2/MaximumMaximumnormalization_2/Sqrt:y:0"normalization_2/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_2/Maximum�
normalization_2/truedivRealDivnormalization_2/sub:z:0normalization_2/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_2/truediv�
&normalization_3/Reshape/ReadVariableOpReadVariableOp/normalization_3_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_3/Reshape/ReadVariableOp�
normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_3/Reshape/shape�
normalization_3/ReshapeReshape.normalization_3/Reshape/ReadVariableOp:value:0&normalization_3/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_3/Reshape�
(normalization_3/Reshape_1/ReadVariableOpReadVariableOp1normalization_3_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_3/Reshape_1/ReadVariableOp�
normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_3/Reshape_1/shape�
normalization_3/Reshape_1Reshape0normalization_3/Reshape_1/ReadVariableOp:value:0(normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_3/Reshape_1�
normalization_3/subSubinputs_2 normalization_3/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_3/sub�
normalization_3/SqrtSqrt"normalization_3/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_3/Maximum/y�
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_3/Maximum�
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_3/truediv�
&normalization_4/Reshape/ReadVariableOpReadVariableOp/normalization_4_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_4/Reshape/ReadVariableOp�
normalization_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_4/Reshape/shape�
normalization_4/ReshapeReshape.normalization_4/Reshape/ReadVariableOp:value:0&normalization_4/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_4/Reshape�
(normalization_4/Reshape_1/ReadVariableOpReadVariableOp1normalization_4_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_4/Reshape_1/ReadVariableOp�
normalization_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_4/Reshape_1/shape�
normalization_4/Reshape_1Reshape0normalization_4/Reshape_1/ReadVariableOp:value:0(normalization_4/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_4/Reshape_1�
normalization_4/subSubinputs_3 normalization_4/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_4/sub�
normalization_4/SqrtSqrt"normalization_4/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_4/Sqrt{
normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_4/Maximum/y�
normalization_4/MaximumMaximumnormalization_4/Sqrt:y:0"normalization_4/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_4/Maximum�
normalization_4/truedivRealDivnormalization_4/sub:z:0normalization_4/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_4/truediv�
&normalization_5/Reshape/ReadVariableOpReadVariableOp/normalization_5_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_5/Reshape/ReadVariableOp�
normalization_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_5/Reshape/shape�
normalization_5/ReshapeReshape.normalization_5/Reshape/ReadVariableOp:value:0&normalization_5/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_5/Reshape�
(normalization_5/Reshape_1/ReadVariableOpReadVariableOp1normalization_5_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_5/Reshape_1/ReadVariableOp�
normalization_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_5/Reshape_1/shape�
normalization_5/Reshape_1Reshape0normalization_5/Reshape_1/ReadVariableOp:value:0(normalization_5/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_5/Reshape_1�
normalization_5/subSubinputs_4 normalization_5/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_5/sub�
normalization_5/SqrtSqrt"normalization_5/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_5/Sqrt{
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_5/Maximum/y�
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_5/Maximum�
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_5/truediv�
&normalization_6/Reshape/ReadVariableOpReadVariableOp/normalization_6_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_6/Reshape/ReadVariableOp�
normalization_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_6/Reshape/shape�
normalization_6/ReshapeReshape.normalization_6/Reshape/ReadVariableOp:value:0&normalization_6/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_6/Reshape�
(normalization_6/Reshape_1/ReadVariableOpReadVariableOp1normalization_6_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_6/Reshape_1/ReadVariableOp�
normalization_6/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_6/Reshape_1/shape�
normalization_6/Reshape_1Reshape0normalization_6/Reshape_1/ReadVariableOp:value:0(normalization_6/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_6/Reshape_1�
normalization_6/subSubinputs_5 normalization_6/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_6/sub�
normalization_6/SqrtSqrt"normalization_6/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_6/Sqrt{
normalization_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_6/Maximum/y�
normalization_6/MaximumMaximumnormalization_6/Sqrt:y:0"normalization_6/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_6/Maximum�
normalization_6/truedivRealDivnormalization_6/sub:z:0normalization_6/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_6/truediv�
&normalization_7/Reshape/ReadVariableOpReadVariableOp/normalization_7_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_7/Reshape/ReadVariableOp�
normalization_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_7/Reshape/shape�
normalization_7/ReshapeReshape.normalization_7/Reshape/ReadVariableOp:value:0&normalization_7/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_7/Reshape�
(normalization_7/Reshape_1/ReadVariableOpReadVariableOp1normalization_7_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_7/Reshape_1/ReadVariableOp�
normalization_7/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_7/Reshape_1/shape�
normalization_7/Reshape_1Reshape0normalization_7/Reshape_1/ReadVariableOp:value:0(normalization_7/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_7/Reshape_1�
normalization_7/subSubinputs_6 normalization_7/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_7/sub�
normalization_7/SqrtSqrt"normalization_7/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_7/Sqrt{
normalization_7/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_7/Maximum/y�
normalization_7/MaximumMaximumnormalization_7/Sqrt:y:0"normalization_7/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_7/Maximum�
normalization_7/truedivRealDivnormalization_7/sub:z:0normalization_7/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_7/truediv�
concatenate/PartitionedCallPartitionedCallnormalization_1/truediv:z:0normalization_2/truediv:z:0normalization_3/truediv:z:0normalization_4/truediv:z:0normalization_5/truediv:z:0normalization_6/truediv:z:0normalization_7/truediv:z:0*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_3193092
concatenate/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_1_319323dense_1_319325*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3193222!
dense_1/StatefulPartitionedCall�
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_3193332
dropout_1/PartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_319346dense_2_319348*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3193452!
dense_2/StatefulPartitionedCall�
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall'^normalization_1/Reshape/ReadVariableOp)^normalization_1/Reshape_1/ReadVariableOp'^normalization_2/Reshape/ReadVariableOp)^normalization_2/Reshape_1/ReadVariableOp'^normalization_3/Reshape/ReadVariableOp)^normalization_3/Reshape_1/ReadVariableOp'^normalization_4/Reshape/ReadVariableOp)^normalization_4/Reshape_1/ReadVariableOp'^normalization_5/Reshape/ReadVariableOp)^normalization_5/Reshape_1/ReadVariableOp'^normalization_6/Reshape/ReadVariableOp)^normalization_6/Reshape_1/ReadVariableOp'^normalization_7/Reshape/ReadVariableOp)^normalization_7/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������: : : : : : : : : : : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2P
&normalization_1/Reshape/ReadVariableOp&normalization_1/Reshape/ReadVariableOp2T
(normalization_1/Reshape_1/ReadVariableOp(normalization_1/Reshape_1/ReadVariableOp2P
&normalization_2/Reshape/ReadVariableOp&normalization_2/Reshape/ReadVariableOp2T
(normalization_2/Reshape_1/ReadVariableOp(normalization_2/Reshape_1/ReadVariableOp2P
&normalization_3/Reshape/ReadVariableOp&normalization_3/Reshape/ReadVariableOp2T
(normalization_3/Reshape_1/ReadVariableOp(normalization_3/Reshape_1/ReadVariableOp2P
&normalization_4/Reshape/ReadVariableOp&normalization_4/Reshape/ReadVariableOp2T
(normalization_4/Reshape_1/ReadVariableOp(normalization_4/Reshape_1/ReadVariableOp2P
&normalization_5/Reshape/ReadVariableOp&normalization_5/Reshape/ReadVariableOp2T
(normalization_5/Reshape_1/ReadVariableOp(normalization_5/Reshape_1/ReadVariableOp2P
&normalization_6/Reshape/ReadVariableOp&normalization_6/Reshape/ReadVariableOp2T
(normalization_6/Reshape_1/ReadVariableOp(normalization_6/Reshape_1/ReadVariableOp2P
&normalization_7/Reshape/ReadVariableOp&normalization_7/Reshape/ReadVariableOp2T
(normalization_7/Reshape_1/ReadVariableOp(normalization_7/Reshape_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_dense_2_layer_call_fn_320738

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3193452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�.
�
__inference_adapt_step_320463
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�IteratorGetNext�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�add/ReadVariableOp�
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:���������*"
output_shapes
:���������*
output_types
22
IteratorGetNextb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsIteratorGetNext:components:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������2

ExpandDimsj
CastCastExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance}
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1V
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	2
Shapeu
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB"       2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis�
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:2

GatherV2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstX
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: 2
Prod|
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	2
add/ReadVariableOp_
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: 2
addW
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_1Q
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_2V
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: 2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/xO
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: 2
subp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpS
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: 2
mul]
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: 2
mul_1L
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: 2
add_1t
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1[
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: 2
sub_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yM
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: 2
powv
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2[
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: 2
add_2J
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: 2
mul_2[
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: 2
sub_2W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yS
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1_
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: 2
add_3N
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: 2
mul_3N
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: 2
add_4�
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignVariableOp�
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype02
AssignVariableOp_1�
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOp_2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
�	
�
C__inference_dense_2_layer_call_and_return_conditional_losses_319345

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�.
�
__inference_adapt_step_320561
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�IteratorGetNext�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�add/ReadVariableOp�
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:���������*"
output_shapes
:���������*
output_types
22
IteratorGetNextb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsIteratorGetNext:components:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������2

ExpandDimsj
CastCastExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance}
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1V
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	2
Shapeu
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB"       2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis�
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:2

GatherV2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstX
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: 2
Prod|
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	2
add/ReadVariableOp_
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: 2
addW
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_1Q
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_2V
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: 2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/xO
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: 2
subp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpS
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: 2
mul]
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: 2
mul_1L
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: 2
add_1t
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1[
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: 2
sub_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yM
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: 2
powv
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2[
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: 2
add_2J
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: 2
mul_2[
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: 2
sub_2W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yS
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1_
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: 2
add_3N
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: 2
mul_3N
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: 2
add_4�
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignVariableOp�
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype02
AssignVariableOp_1�
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOp_2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
�
F
*__inference_dropout_1_layer_call_fn_320707

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_3193332
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
!__inference__wrapped_model_319187	
slope

aspect
	elevation
twi
curv
lito
uso_solo?
5model_normalization_1_reshape_readvariableop_resource: A
7model_normalization_1_reshape_1_readvariableop_resource: ?
5model_normalization_2_reshape_readvariableop_resource: A
7model_normalization_2_reshape_1_readvariableop_resource: ?
5model_normalization_3_reshape_readvariableop_resource: A
7model_normalization_3_reshape_1_readvariableop_resource: ?
5model_normalization_4_reshape_readvariableop_resource: A
7model_normalization_4_reshape_1_readvariableop_resource: ?
5model_normalization_5_reshape_readvariableop_resource: A
7model_normalization_5_reshape_1_readvariableop_resource: ?
5model_normalization_6_reshape_readvariableop_resource: A
7model_normalization_6_reshape_1_readvariableop_resource: ?
5model_normalization_7_reshape_readvariableop_resource: A
7model_normalization_7_reshape_1_readvariableop_resource: >
,model_dense_1_matmul_readvariableop_resource:;
-model_dense_1_biasadd_readvariableop_resource:>
,model_dense_2_matmul_readvariableop_resource:;
-model_dense_2_biasadd_readvariableop_resource:
identity��$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�$model/dense_2/BiasAdd/ReadVariableOp�#model/dense_2/MatMul/ReadVariableOp�,model/normalization_1/Reshape/ReadVariableOp�.model/normalization_1/Reshape_1/ReadVariableOp�,model/normalization_2/Reshape/ReadVariableOp�.model/normalization_2/Reshape_1/ReadVariableOp�,model/normalization_3/Reshape/ReadVariableOp�.model/normalization_3/Reshape_1/ReadVariableOp�,model/normalization_4/Reshape/ReadVariableOp�.model/normalization_4/Reshape_1/ReadVariableOp�,model/normalization_5/Reshape/ReadVariableOp�.model/normalization_5/Reshape_1/ReadVariableOp�,model/normalization_6/Reshape/ReadVariableOp�.model/normalization_6/Reshape_1/ReadVariableOp�,model/normalization_7/Reshape/ReadVariableOp�.model/normalization_7/Reshape_1/ReadVariableOp�
,model/normalization_1/Reshape/ReadVariableOpReadVariableOp5model_normalization_1_reshape_readvariableop_resource*
_output_shapes
: *
dtype02.
,model/normalization_1/Reshape/ReadVariableOp�
#model/normalization_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2%
#model/normalization_1/Reshape/shape�
model/normalization_1/ReshapeReshape4model/normalization_1/Reshape/ReadVariableOp:value:0,model/normalization_1/Reshape/shape:output:0*
T0*
_output_shapes

:2
model/normalization_1/Reshape�
.model/normalization_1/Reshape_1/ReadVariableOpReadVariableOp7model_normalization_1_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype020
.model/normalization_1/Reshape_1/ReadVariableOp�
%model/normalization_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2'
%model/normalization_1/Reshape_1/shape�
model/normalization_1/Reshape_1Reshape6model/normalization_1/Reshape_1/ReadVariableOp:value:0.model/normalization_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:2!
model/normalization_1/Reshape_1�
model/normalization_1/subSubslope&model/normalization_1/Reshape:output:0*
T0*'
_output_shapes
:���������2
model/normalization_1/sub�
model/normalization_1/SqrtSqrt(model/normalization_1/Reshape_1:output:0*
T0*
_output_shapes

:2
model/normalization_1/Sqrt�
model/normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32!
model/normalization_1/Maximum/y�
model/normalization_1/MaximumMaximummodel/normalization_1/Sqrt:y:0(model/normalization_1/Maximum/y:output:0*
T0*
_output_shapes

:2
model/normalization_1/Maximum�
model/normalization_1/truedivRealDivmodel/normalization_1/sub:z:0!model/normalization_1/Maximum:z:0*
T0*'
_output_shapes
:���������2
model/normalization_1/truediv�
,model/normalization_2/Reshape/ReadVariableOpReadVariableOp5model_normalization_2_reshape_readvariableop_resource*
_output_shapes
: *
dtype02.
,model/normalization_2/Reshape/ReadVariableOp�
#model/normalization_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2%
#model/normalization_2/Reshape/shape�
model/normalization_2/ReshapeReshape4model/normalization_2/Reshape/ReadVariableOp:value:0,model/normalization_2/Reshape/shape:output:0*
T0*
_output_shapes

:2
model/normalization_2/Reshape�
.model/normalization_2/Reshape_1/ReadVariableOpReadVariableOp7model_normalization_2_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype020
.model/normalization_2/Reshape_1/ReadVariableOp�
%model/normalization_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2'
%model/normalization_2/Reshape_1/shape�
model/normalization_2/Reshape_1Reshape6model/normalization_2/Reshape_1/ReadVariableOp:value:0.model/normalization_2/Reshape_1/shape:output:0*
T0*
_output_shapes

:2!
model/normalization_2/Reshape_1�
model/normalization_2/subSubaspect&model/normalization_2/Reshape:output:0*
T0*'
_output_shapes
:���������2
model/normalization_2/sub�
model/normalization_2/SqrtSqrt(model/normalization_2/Reshape_1:output:0*
T0*
_output_shapes

:2
model/normalization_2/Sqrt�
model/normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32!
model/normalization_2/Maximum/y�
model/normalization_2/MaximumMaximummodel/normalization_2/Sqrt:y:0(model/normalization_2/Maximum/y:output:0*
T0*
_output_shapes

:2
model/normalization_2/Maximum�
model/normalization_2/truedivRealDivmodel/normalization_2/sub:z:0!model/normalization_2/Maximum:z:0*
T0*'
_output_shapes
:���������2
model/normalization_2/truediv�
,model/normalization_3/Reshape/ReadVariableOpReadVariableOp5model_normalization_3_reshape_readvariableop_resource*
_output_shapes
: *
dtype02.
,model/normalization_3/Reshape/ReadVariableOp�
#model/normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2%
#model/normalization_3/Reshape/shape�
model/normalization_3/ReshapeReshape4model/normalization_3/Reshape/ReadVariableOp:value:0,model/normalization_3/Reshape/shape:output:0*
T0*
_output_shapes

:2
model/normalization_3/Reshape�
.model/normalization_3/Reshape_1/ReadVariableOpReadVariableOp7model_normalization_3_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype020
.model/normalization_3/Reshape_1/ReadVariableOp�
%model/normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2'
%model/normalization_3/Reshape_1/shape�
model/normalization_3/Reshape_1Reshape6model/normalization_3/Reshape_1/ReadVariableOp:value:0.model/normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes

:2!
model/normalization_3/Reshape_1�
model/normalization_3/subSub	elevation&model/normalization_3/Reshape:output:0*
T0*'
_output_shapes
:���������2
model/normalization_3/sub�
model/normalization_3/SqrtSqrt(model/normalization_3/Reshape_1:output:0*
T0*
_output_shapes

:2
model/normalization_3/Sqrt�
model/normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32!
model/normalization_3/Maximum/y�
model/normalization_3/MaximumMaximummodel/normalization_3/Sqrt:y:0(model/normalization_3/Maximum/y:output:0*
T0*
_output_shapes

:2
model/normalization_3/Maximum�
model/normalization_3/truedivRealDivmodel/normalization_3/sub:z:0!model/normalization_3/Maximum:z:0*
T0*'
_output_shapes
:���������2
model/normalization_3/truediv�
,model/normalization_4/Reshape/ReadVariableOpReadVariableOp5model_normalization_4_reshape_readvariableop_resource*
_output_shapes
: *
dtype02.
,model/normalization_4/Reshape/ReadVariableOp�
#model/normalization_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2%
#model/normalization_4/Reshape/shape�
model/normalization_4/ReshapeReshape4model/normalization_4/Reshape/ReadVariableOp:value:0,model/normalization_4/Reshape/shape:output:0*
T0*
_output_shapes

:2
model/normalization_4/Reshape�
.model/normalization_4/Reshape_1/ReadVariableOpReadVariableOp7model_normalization_4_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype020
.model/normalization_4/Reshape_1/ReadVariableOp�
%model/normalization_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2'
%model/normalization_4/Reshape_1/shape�
model/normalization_4/Reshape_1Reshape6model/normalization_4/Reshape_1/ReadVariableOp:value:0.model/normalization_4/Reshape_1/shape:output:0*
T0*
_output_shapes

:2!
model/normalization_4/Reshape_1�
model/normalization_4/subSubtwi&model/normalization_4/Reshape:output:0*
T0*'
_output_shapes
:���������2
model/normalization_4/sub�
model/normalization_4/SqrtSqrt(model/normalization_4/Reshape_1:output:0*
T0*
_output_shapes

:2
model/normalization_4/Sqrt�
model/normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32!
model/normalization_4/Maximum/y�
model/normalization_4/MaximumMaximummodel/normalization_4/Sqrt:y:0(model/normalization_4/Maximum/y:output:0*
T0*
_output_shapes

:2
model/normalization_4/Maximum�
model/normalization_4/truedivRealDivmodel/normalization_4/sub:z:0!model/normalization_4/Maximum:z:0*
T0*'
_output_shapes
:���������2
model/normalization_4/truediv�
,model/normalization_5/Reshape/ReadVariableOpReadVariableOp5model_normalization_5_reshape_readvariableop_resource*
_output_shapes
: *
dtype02.
,model/normalization_5/Reshape/ReadVariableOp�
#model/normalization_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2%
#model/normalization_5/Reshape/shape�
model/normalization_5/ReshapeReshape4model/normalization_5/Reshape/ReadVariableOp:value:0,model/normalization_5/Reshape/shape:output:0*
T0*
_output_shapes

:2
model/normalization_5/Reshape�
.model/normalization_5/Reshape_1/ReadVariableOpReadVariableOp7model_normalization_5_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype020
.model/normalization_5/Reshape_1/ReadVariableOp�
%model/normalization_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2'
%model/normalization_5/Reshape_1/shape�
model/normalization_5/Reshape_1Reshape6model/normalization_5/Reshape_1/ReadVariableOp:value:0.model/normalization_5/Reshape_1/shape:output:0*
T0*
_output_shapes

:2!
model/normalization_5/Reshape_1�
model/normalization_5/subSubcurv&model/normalization_5/Reshape:output:0*
T0*'
_output_shapes
:���������2
model/normalization_5/sub�
model/normalization_5/SqrtSqrt(model/normalization_5/Reshape_1:output:0*
T0*
_output_shapes

:2
model/normalization_5/Sqrt�
model/normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32!
model/normalization_5/Maximum/y�
model/normalization_5/MaximumMaximummodel/normalization_5/Sqrt:y:0(model/normalization_5/Maximum/y:output:0*
T0*
_output_shapes

:2
model/normalization_5/Maximum�
model/normalization_5/truedivRealDivmodel/normalization_5/sub:z:0!model/normalization_5/Maximum:z:0*
T0*'
_output_shapes
:���������2
model/normalization_5/truediv�
,model/normalization_6/Reshape/ReadVariableOpReadVariableOp5model_normalization_6_reshape_readvariableop_resource*
_output_shapes
: *
dtype02.
,model/normalization_6/Reshape/ReadVariableOp�
#model/normalization_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2%
#model/normalization_6/Reshape/shape�
model/normalization_6/ReshapeReshape4model/normalization_6/Reshape/ReadVariableOp:value:0,model/normalization_6/Reshape/shape:output:0*
T0*
_output_shapes

:2
model/normalization_6/Reshape�
.model/normalization_6/Reshape_1/ReadVariableOpReadVariableOp7model_normalization_6_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype020
.model/normalization_6/Reshape_1/ReadVariableOp�
%model/normalization_6/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2'
%model/normalization_6/Reshape_1/shape�
model/normalization_6/Reshape_1Reshape6model/normalization_6/Reshape_1/ReadVariableOp:value:0.model/normalization_6/Reshape_1/shape:output:0*
T0*
_output_shapes

:2!
model/normalization_6/Reshape_1�
model/normalization_6/subSublito&model/normalization_6/Reshape:output:0*
T0*'
_output_shapes
:���������2
model/normalization_6/sub�
model/normalization_6/SqrtSqrt(model/normalization_6/Reshape_1:output:0*
T0*
_output_shapes

:2
model/normalization_6/Sqrt�
model/normalization_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32!
model/normalization_6/Maximum/y�
model/normalization_6/MaximumMaximummodel/normalization_6/Sqrt:y:0(model/normalization_6/Maximum/y:output:0*
T0*
_output_shapes

:2
model/normalization_6/Maximum�
model/normalization_6/truedivRealDivmodel/normalization_6/sub:z:0!model/normalization_6/Maximum:z:0*
T0*'
_output_shapes
:���������2
model/normalization_6/truediv�
,model/normalization_7/Reshape/ReadVariableOpReadVariableOp5model_normalization_7_reshape_readvariableop_resource*
_output_shapes
: *
dtype02.
,model/normalization_7/Reshape/ReadVariableOp�
#model/normalization_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2%
#model/normalization_7/Reshape/shape�
model/normalization_7/ReshapeReshape4model/normalization_7/Reshape/ReadVariableOp:value:0,model/normalization_7/Reshape/shape:output:0*
T0*
_output_shapes

:2
model/normalization_7/Reshape�
.model/normalization_7/Reshape_1/ReadVariableOpReadVariableOp7model_normalization_7_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype020
.model/normalization_7/Reshape_1/ReadVariableOp�
%model/normalization_7/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2'
%model/normalization_7/Reshape_1/shape�
model/normalization_7/Reshape_1Reshape6model/normalization_7/Reshape_1/ReadVariableOp:value:0.model/normalization_7/Reshape_1/shape:output:0*
T0*
_output_shapes

:2!
model/normalization_7/Reshape_1�
model/normalization_7/subSubuso_solo&model/normalization_7/Reshape:output:0*
T0*'
_output_shapes
:���������2
model/normalization_7/sub�
model/normalization_7/SqrtSqrt(model/normalization_7/Reshape_1:output:0*
T0*
_output_shapes

:2
model/normalization_7/Sqrt�
model/normalization_7/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32!
model/normalization_7/Maximum/y�
model/normalization_7/MaximumMaximummodel/normalization_7/Sqrt:y:0(model/normalization_7/Maximum/y:output:0*
T0*
_output_shapes

:2
model/normalization_7/Maximum�
model/normalization_7/truedivRealDivmodel/normalization_7/sub:z:0!model/normalization_7/Maximum:z:0*
T0*'
_output_shapes
:���������2
model/normalization_7/truediv�
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis�
model/concatenate/concatConcatV2!model/normalization_1/truediv:z:0!model/normalization_2/truediv:z:0!model/normalization_3/truediv:z:0!model/normalization_4/truediv:z:0!model/normalization_5/truediv:z:0!model/normalization_6/truediv:z:0!model/normalization_7/truediv:z:0&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
model/concatenate/concat�
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#model/dense_1/MatMul/ReadVariableOp�
model/dense_1/MatMulMatMul!model/concatenate/concat:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/dense_1/MatMul�
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/dense_1/BiasAdd�
model/dense_1/TanhTanhmodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
model/dense_1/Tanh�
model/dropout_1/IdentityIdentitymodel/dense_1/Tanh:y:0*
T0*'
_output_shapes
:���������2
model/dropout_1/Identity�
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#model/dense_2/MatMul/ReadVariableOp�
model/dense_2/MatMulMatMul!model/dropout_1/Identity:output:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/dense_2/MatMul�
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_2/BiasAdd/ReadVariableOp�
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/dense_2/BiasAdd�
IdentityIdentitymodel/dense_2/BiasAdd:output:0%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp-^model/normalization_1/Reshape/ReadVariableOp/^model/normalization_1/Reshape_1/ReadVariableOp-^model/normalization_2/Reshape/ReadVariableOp/^model/normalization_2/Reshape_1/ReadVariableOp-^model/normalization_3/Reshape/ReadVariableOp/^model/normalization_3/Reshape_1/ReadVariableOp-^model/normalization_4/Reshape/ReadVariableOp/^model/normalization_4/Reshape_1/ReadVariableOp-^model/normalization_5/Reshape/ReadVariableOp/^model/normalization_5/Reshape_1/ReadVariableOp-^model/normalization_6/Reshape/ReadVariableOp/^model/normalization_6/Reshape_1/ReadVariableOp-^model/normalization_7/Reshape/ReadVariableOp/^model/normalization_7/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������: : : : : : : : : : : : : : : : : : 2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2\
,model/normalization_1/Reshape/ReadVariableOp,model/normalization_1/Reshape/ReadVariableOp2`
.model/normalization_1/Reshape_1/ReadVariableOp.model/normalization_1/Reshape_1/ReadVariableOp2\
,model/normalization_2/Reshape/ReadVariableOp,model/normalization_2/Reshape/ReadVariableOp2`
.model/normalization_2/Reshape_1/ReadVariableOp.model/normalization_2/Reshape_1/ReadVariableOp2\
,model/normalization_3/Reshape/ReadVariableOp,model/normalization_3/Reshape/ReadVariableOp2`
.model/normalization_3/Reshape_1/ReadVariableOp.model/normalization_3/Reshape_1/ReadVariableOp2\
,model/normalization_4/Reshape/ReadVariableOp,model/normalization_4/Reshape/ReadVariableOp2`
.model/normalization_4/Reshape_1/ReadVariableOp.model/normalization_4/Reshape_1/ReadVariableOp2\
,model/normalization_5/Reshape/ReadVariableOp,model/normalization_5/Reshape/ReadVariableOp2`
.model/normalization_5/Reshape_1/ReadVariableOp.model/normalization_5/Reshape_1/ReadVariableOp2\
,model/normalization_6/Reshape/ReadVariableOp,model/normalization_6/Reshape/ReadVariableOp2`
.model/normalization_6/Reshape_1/ReadVariableOp.model/normalization_6/Reshape_1/ReadVariableOp2\
,model/normalization_7/Reshape/ReadVariableOp,model/normalization_7/Reshape/ReadVariableOp2`
.model/normalization_7/Reshape_1/ReadVariableOp.model/normalization_7/Reshape_1/ReadVariableOp:N J
'
_output_shapes
:���������

_user_specified_nameslope:OK
'
_output_shapes
:���������
 
_user_specified_nameaspect:RN
'
_output_shapes
:���������
#
_user_specified_name	elevation:LH
'
_output_shapes
:���������

_user_specified_nametwi:MI
'
_output_shapes
:���������

_user_specified_namecurv:MI
'
_output_shapes
:���������

_user_specified_namelito:QM
'
_output_shapes
:���������
"
_user_specified_name
uso_solo
�
�
(__inference_dense_1_layer_call_fn_320691

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3193222
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
C__inference_dense_1_layer_call_and_return_conditional_losses_320702

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
C__inference_dense_2_layer_call_and_return_conditional_losses_320748

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�.
�
__inference_adapt_step_320659
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�IteratorGetNext�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�add/ReadVariableOp�
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:���������*"
output_shapes
:���������*
output_types
2	2
IteratorGetNextb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsIteratorGetNext:components:0ExpandDims/dim:output:0*
T0	*'
_output_shapes
:���������2

ExpandDimsj
CastCastExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:���������2
Cast�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance}
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1V
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	2
Shapeu
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB"       2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis�
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:2

GatherV2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstX
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: 2
Prod|
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	2
add/ReadVariableOp_
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: 2
addW
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_1Q
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_2V
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: 2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/xO
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: 2
subp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpS
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: 2
mul]
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: 2
mul_1L
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: 2
add_1t
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1[
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: 2
sub_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yM
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: 2
powv
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2[
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: 2
add_2J
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: 2
mul_2[
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: 2
sub_2W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yS
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1_
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: 2
add_3N
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: 2
mul_3N
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: 2
add_4�
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignVariableOp�
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype02
AssignVariableOp_1�
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOp_2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
��
�
A__inference_model_layer_call_and_return_conditional_losses_319614

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_69
/normalization_1_reshape_readvariableop_resource: ;
1normalization_1_reshape_1_readvariableop_resource: 9
/normalization_2_reshape_readvariableop_resource: ;
1normalization_2_reshape_1_readvariableop_resource: 9
/normalization_3_reshape_readvariableop_resource: ;
1normalization_3_reshape_1_readvariableop_resource: 9
/normalization_4_reshape_readvariableop_resource: ;
1normalization_4_reshape_1_readvariableop_resource: 9
/normalization_5_reshape_readvariableop_resource: ;
1normalization_5_reshape_1_readvariableop_resource: 9
/normalization_6_reshape_readvariableop_resource: ;
1normalization_6_reshape_1_readvariableop_resource: 9
/normalization_7_reshape_readvariableop_resource: ;
1normalization_7_reshape_1_readvariableop_resource:  
dense_1_319602:
dense_1_319604: 
dense_2_319608:
dense_2_319610:
identity��dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�&normalization_1/Reshape/ReadVariableOp�(normalization_1/Reshape_1/ReadVariableOp�&normalization_2/Reshape/ReadVariableOp�(normalization_2/Reshape_1/ReadVariableOp�&normalization_3/Reshape/ReadVariableOp�(normalization_3/Reshape_1/ReadVariableOp�&normalization_4/Reshape/ReadVariableOp�(normalization_4/Reshape_1/ReadVariableOp�&normalization_5/Reshape/ReadVariableOp�(normalization_5/Reshape_1/ReadVariableOp�&normalization_6/Reshape/ReadVariableOp�(normalization_6/Reshape_1/ReadVariableOp�&normalization_7/Reshape/ReadVariableOp�(normalization_7/Reshape_1/ReadVariableOp�
&normalization_1/Reshape/ReadVariableOpReadVariableOp/normalization_1_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_1/Reshape/ReadVariableOp�
normalization_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_1/Reshape/shape�
normalization_1/ReshapeReshape.normalization_1/Reshape/ReadVariableOp:value:0&normalization_1/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_1/Reshape�
(normalization_1/Reshape_1/ReadVariableOpReadVariableOp1normalization_1_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_1/Reshape_1/ReadVariableOp�
normalization_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_1/Reshape_1/shape�
normalization_1/Reshape_1Reshape0normalization_1/Reshape_1/ReadVariableOp:value:0(normalization_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_1/Reshape_1�
normalization_1/subSubinputs normalization_1/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_1/sub�
normalization_1/SqrtSqrt"normalization_1/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_1/Sqrt{
normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_1/Maximum/y�
normalization_1/MaximumMaximumnormalization_1/Sqrt:y:0"normalization_1/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_1/Maximum�
normalization_1/truedivRealDivnormalization_1/sub:z:0normalization_1/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_1/truediv�
&normalization_2/Reshape/ReadVariableOpReadVariableOp/normalization_2_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_2/Reshape/ReadVariableOp�
normalization_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_2/Reshape/shape�
normalization_2/ReshapeReshape.normalization_2/Reshape/ReadVariableOp:value:0&normalization_2/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_2/Reshape�
(normalization_2/Reshape_1/ReadVariableOpReadVariableOp1normalization_2_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_2/Reshape_1/ReadVariableOp�
normalization_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_2/Reshape_1/shape�
normalization_2/Reshape_1Reshape0normalization_2/Reshape_1/ReadVariableOp:value:0(normalization_2/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_2/Reshape_1�
normalization_2/subSubinputs_1 normalization_2/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_2/sub�
normalization_2/SqrtSqrt"normalization_2/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_2/Sqrt{
normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_2/Maximum/y�
normalization_2/MaximumMaximumnormalization_2/Sqrt:y:0"normalization_2/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_2/Maximum�
normalization_2/truedivRealDivnormalization_2/sub:z:0normalization_2/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_2/truediv�
&normalization_3/Reshape/ReadVariableOpReadVariableOp/normalization_3_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_3/Reshape/ReadVariableOp�
normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_3/Reshape/shape�
normalization_3/ReshapeReshape.normalization_3/Reshape/ReadVariableOp:value:0&normalization_3/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_3/Reshape�
(normalization_3/Reshape_1/ReadVariableOpReadVariableOp1normalization_3_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_3/Reshape_1/ReadVariableOp�
normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_3/Reshape_1/shape�
normalization_3/Reshape_1Reshape0normalization_3/Reshape_1/ReadVariableOp:value:0(normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_3/Reshape_1�
normalization_3/subSubinputs_2 normalization_3/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_3/sub�
normalization_3/SqrtSqrt"normalization_3/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_3/Maximum/y�
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_3/Maximum�
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_3/truediv�
&normalization_4/Reshape/ReadVariableOpReadVariableOp/normalization_4_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_4/Reshape/ReadVariableOp�
normalization_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_4/Reshape/shape�
normalization_4/ReshapeReshape.normalization_4/Reshape/ReadVariableOp:value:0&normalization_4/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_4/Reshape�
(normalization_4/Reshape_1/ReadVariableOpReadVariableOp1normalization_4_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_4/Reshape_1/ReadVariableOp�
normalization_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_4/Reshape_1/shape�
normalization_4/Reshape_1Reshape0normalization_4/Reshape_1/ReadVariableOp:value:0(normalization_4/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_4/Reshape_1�
normalization_4/subSubinputs_3 normalization_4/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_4/sub�
normalization_4/SqrtSqrt"normalization_4/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_4/Sqrt{
normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_4/Maximum/y�
normalization_4/MaximumMaximumnormalization_4/Sqrt:y:0"normalization_4/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_4/Maximum�
normalization_4/truedivRealDivnormalization_4/sub:z:0normalization_4/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_4/truediv�
&normalization_5/Reshape/ReadVariableOpReadVariableOp/normalization_5_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_5/Reshape/ReadVariableOp�
normalization_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_5/Reshape/shape�
normalization_5/ReshapeReshape.normalization_5/Reshape/ReadVariableOp:value:0&normalization_5/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_5/Reshape�
(normalization_5/Reshape_1/ReadVariableOpReadVariableOp1normalization_5_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_5/Reshape_1/ReadVariableOp�
normalization_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_5/Reshape_1/shape�
normalization_5/Reshape_1Reshape0normalization_5/Reshape_1/ReadVariableOp:value:0(normalization_5/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_5/Reshape_1�
normalization_5/subSubinputs_4 normalization_5/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_5/sub�
normalization_5/SqrtSqrt"normalization_5/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_5/Sqrt{
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_5/Maximum/y�
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_5/Maximum�
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_5/truediv�
&normalization_6/Reshape/ReadVariableOpReadVariableOp/normalization_6_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_6/Reshape/ReadVariableOp�
normalization_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_6/Reshape/shape�
normalization_6/ReshapeReshape.normalization_6/Reshape/ReadVariableOp:value:0&normalization_6/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_6/Reshape�
(normalization_6/Reshape_1/ReadVariableOpReadVariableOp1normalization_6_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_6/Reshape_1/ReadVariableOp�
normalization_6/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_6/Reshape_1/shape�
normalization_6/Reshape_1Reshape0normalization_6/Reshape_1/ReadVariableOp:value:0(normalization_6/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_6/Reshape_1�
normalization_6/subSubinputs_5 normalization_6/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_6/sub�
normalization_6/SqrtSqrt"normalization_6/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_6/Sqrt{
normalization_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_6/Maximum/y�
normalization_6/MaximumMaximumnormalization_6/Sqrt:y:0"normalization_6/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_6/Maximum�
normalization_6/truedivRealDivnormalization_6/sub:z:0normalization_6/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_6/truediv�
&normalization_7/Reshape/ReadVariableOpReadVariableOp/normalization_7_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_7/Reshape/ReadVariableOp�
normalization_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_7/Reshape/shape�
normalization_7/ReshapeReshape.normalization_7/Reshape/ReadVariableOp:value:0&normalization_7/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_7/Reshape�
(normalization_7/Reshape_1/ReadVariableOpReadVariableOp1normalization_7_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_7/Reshape_1/ReadVariableOp�
normalization_7/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_7/Reshape_1/shape�
normalization_7/Reshape_1Reshape0normalization_7/Reshape_1/ReadVariableOp:value:0(normalization_7/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_7/Reshape_1�
normalization_7/subSubinputs_6 normalization_7/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_7/sub�
normalization_7/SqrtSqrt"normalization_7/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_7/Sqrt{
normalization_7/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_7/Maximum/y�
normalization_7/MaximumMaximumnormalization_7/Sqrt:y:0"normalization_7/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_7/Maximum�
normalization_7/truedivRealDivnormalization_7/sub:z:0normalization_7/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_7/truediv�
concatenate/PartitionedCallPartitionedCallnormalization_1/truediv:z:0normalization_2/truediv:z:0normalization_3/truediv:z:0normalization_4/truediv:z:0normalization_5/truediv:z:0normalization_6/truediv:z:0normalization_7/truediv:z:0*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_3193092
concatenate/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_1_319602dense_1_319604*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3193222!
dense_1/StatefulPartitionedCall�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_3194212#
!dropout_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_319608dense_2_319610*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3193452!
dense_2/StatefulPartitionedCall�
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall'^normalization_1/Reshape/ReadVariableOp)^normalization_1/Reshape_1/ReadVariableOp'^normalization_2/Reshape/ReadVariableOp)^normalization_2/Reshape_1/ReadVariableOp'^normalization_3/Reshape/ReadVariableOp)^normalization_3/Reshape_1/ReadVariableOp'^normalization_4/Reshape/ReadVariableOp)^normalization_4/Reshape_1/ReadVariableOp'^normalization_5/Reshape/ReadVariableOp)^normalization_5/Reshape_1/ReadVariableOp'^normalization_6/Reshape/ReadVariableOp)^normalization_6/Reshape_1/ReadVariableOp'^normalization_7/Reshape/ReadVariableOp)^normalization_7/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������: : : : : : : : : : : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2P
&normalization_1/Reshape/ReadVariableOp&normalization_1/Reshape/ReadVariableOp2T
(normalization_1/Reshape_1/ReadVariableOp(normalization_1/Reshape_1/ReadVariableOp2P
&normalization_2/Reshape/ReadVariableOp&normalization_2/Reshape/ReadVariableOp2T
(normalization_2/Reshape_1/ReadVariableOp(normalization_2/Reshape_1/ReadVariableOp2P
&normalization_3/Reshape/ReadVariableOp&normalization_3/Reshape/ReadVariableOp2T
(normalization_3/Reshape_1/ReadVariableOp(normalization_3/Reshape_1/ReadVariableOp2P
&normalization_4/Reshape/ReadVariableOp&normalization_4/Reshape/ReadVariableOp2T
(normalization_4/Reshape_1/ReadVariableOp(normalization_4/Reshape_1/ReadVariableOp2P
&normalization_5/Reshape/ReadVariableOp&normalization_5/Reshape/ReadVariableOp2T
(normalization_5/Reshape_1/ReadVariableOp(normalization_5/Reshape_1/ReadVariableOp2P
&normalization_6/Reshape/ReadVariableOp&normalization_6/Reshape/ReadVariableOp2T
(normalization_6/Reshape_1/ReadVariableOp(normalization_6/Reshape_1/ReadVariableOp2P
&normalization_7/Reshape/ReadVariableOp&normalization_7/Reshape/ReadVariableOp2T
(normalization_7/Reshape_1/ReadVariableOp(normalization_7/Reshape_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_319981

aspect
curv
	elevation
lito	
slope
twi
uso_solo
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: 

unknown_13:

unknown_14:

unknown_15:

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallslopeaspect	elevationtwicurvlitouso_solounknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_3191872
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameaspect:MI
'
_output_shapes
:���������

_user_specified_namecurv:RN
'
_output_shapes
:���������
#
_user_specified_name	elevation:MI
'
_output_shapes
:���������

_user_specified_namelito:NJ
'
_output_shapes
:���������

_user_specified_nameslope:LH
'
_output_shapes
:���������

_user_specified_nametwi:QM
'
_output_shapes
:���������
"
_user_specified_name
uso_solo
�.
�
__inference_adapt_step_320512
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�IteratorGetNext�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�add/ReadVariableOp�
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:���������*"
output_shapes
:���������*
output_types
22
IteratorGetNextb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsIteratorGetNext:components:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������2

ExpandDimsj
CastCastExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance}
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1V
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	2
Shapeu
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB"       2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis�
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:2

GatherV2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstX
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: 2
Prod|
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	2
add/ReadVariableOp_
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: 2
addW
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_1Q
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_2V
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: 2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/xO
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: 2
subp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpS
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: 2
mul]
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: 2
mul_1L
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: 2
add_1t
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1[
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: 2
sub_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yM
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: 2
powv
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2[
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: 2
add_2J
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: 2
mul_2[
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: 2
sub_2W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yS
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1_
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: 2
add_3N
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: 2
mul_3N
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: 2
add_4�
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignVariableOp�
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype02
AssignVariableOp_1�
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOp_2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
�

�
C__inference_dense_1_layer_call_and_return_conditional_losses_319322

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_319333

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_model_layer_call_fn_319700	
slope

aspect
	elevation
twi
curv
lito
uso_solo
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: 

unknown_13:

unknown_14:

unknown_15:

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallslopeaspect	elevationtwicurvlitouso_solounknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_3196142
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:���������

_user_specified_nameslope:OK
'
_output_shapes
:���������
 
_user_specified_nameaspect:RN
'
_output_shapes
:���������
#
_user_specified_name	elevation:LH
'
_output_shapes
:���������

_user_specified_nametwi:MI
'
_output_shapes
:���������

_user_specified_namecurv:MI
'
_output_shapes
:���������

_user_specified_namelito:QM
'
_output_shapes
:���������
"
_user_specified_name
uso_solo
�

�
G__inference_concatenate_layer_call_and_return_conditional_losses_319309

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
A__inference_model_layer_call_and_return_conditional_losses_319926	
slope

aspect
	elevation
twi
curv
lito
uso_solo9
/normalization_1_reshape_readvariableop_resource: ;
1normalization_1_reshape_1_readvariableop_resource: 9
/normalization_2_reshape_readvariableop_resource: ;
1normalization_2_reshape_1_readvariableop_resource: 9
/normalization_3_reshape_readvariableop_resource: ;
1normalization_3_reshape_1_readvariableop_resource: 9
/normalization_4_reshape_readvariableop_resource: ;
1normalization_4_reshape_1_readvariableop_resource: 9
/normalization_5_reshape_readvariableop_resource: ;
1normalization_5_reshape_1_readvariableop_resource: 9
/normalization_6_reshape_readvariableop_resource: ;
1normalization_6_reshape_1_readvariableop_resource: 9
/normalization_7_reshape_readvariableop_resource: ;
1normalization_7_reshape_1_readvariableop_resource:  
dense_1_319914:
dense_1_319916: 
dense_2_319920:
dense_2_319922:
identity��dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�&normalization_1/Reshape/ReadVariableOp�(normalization_1/Reshape_1/ReadVariableOp�&normalization_2/Reshape/ReadVariableOp�(normalization_2/Reshape_1/ReadVariableOp�&normalization_3/Reshape/ReadVariableOp�(normalization_3/Reshape_1/ReadVariableOp�&normalization_4/Reshape/ReadVariableOp�(normalization_4/Reshape_1/ReadVariableOp�&normalization_5/Reshape/ReadVariableOp�(normalization_5/Reshape_1/ReadVariableOp�&normalization_6/Reshape/ReadVariableOp�(normalization_6/Reshape_1/ReadVariableOp�&normalization_7/Reshape/ReadVariableOp�(normalization_7/Reshape_1/ReadVariableOp�
&normalization_1/Reshape/ReadVariableOpReadVariableOp/normalization_1_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_1/Reshape/ReadVariableOp�
normalization_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_1/Reshape/shape�
normalization_1/ReshapeReshape.normalization_1/Reshape/ReadVariableOp:value:0&normalization_1/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_1/Reshape�
(normalization_1/Reshape_1/ReadVariableOpReadVariableOp1normalization_1_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_1/Reshape_1/ReadVariableOp�
normalization_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_1/Reshape_1/shape�
normalization_1/Reshape_1Reshape0normalization_1/Reshape_1/ReadVariableOp:value:0(normalization_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_1/Reshape_1�
normalization_1/subSubslope normalization_1/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_1/sub�
normalization_1/SqrtSqrt"normalization_1/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_1/Sqrt{
normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_1/Maximum/y�
normalization_1/MaximumMaximumnormalization_1/Sqrt:y:0"normalization_1/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_1/Maximum�
normalization_1/truedivRealDivnormalization_1/sub:z:0normalization_1/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_1/truediv�
&normalization_2/Reshape/ReadVariableOpReadVariableOp/normalization_2_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_2/Reshape/ReadVariableOp�
normalization_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_2/Reshape/shape�
normalization_2/ReshapeReshape.normalization_2/Reshape/ReadVariableOp:value:0&normalization_2/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_2/Reshape�
(normalization_2/Reshape_1/ReadVariableOpReadVariableOp1normalization_2_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_2/Reshape_1/ReadVariableOp�
normalization_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_2/Reshape_1/shape�
normalization_2/Reshape_1Reshape0normalization_2/Reshape_1/ReadVariableOp:value:0(normalization_2/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_2/Reshape_1�
normalization_2/subSubaspect normalization_2/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_2/sub�
normalization_2/SqrtSqrt"normalization_2/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_2/Sqrt{
normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_2/Maximum/y�
normalization_2/MaximumMaximumnormalization_2/Sqrt:y:0"normalization_2/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_2/Maximum�
normalization_2/truedivRealDivnormalization_2/sub:z:0normalization_2/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_2/truediv�
&normalization_3/Reshape/ReadVariableOpReadVariableOp/normalization_3_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_3/Reshape/ReadVariableOp�
normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_3/Reshape/shape�
normalization_3/ReshapeReshape.normalization_3/Reshape/ReadVariableOp:value:0&normalization_3/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_3/Reshape�
(normalization_3/Reshape_1/ReadVariableOpReadVariableOp1normalization_3_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_3/Reshape_1/ReadVariableOp�
normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_3/Reshape_1/shape�
normalization_3/Reshape_1Reshape0normalization_3/Reshape_1/ReadVariableOp:value:0(normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_3/Reshape_1�
normalization_3/subSub	elevation normalization_3/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_3/sub�
normalization_3/SqrtSqrt"normalization_3/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_3/Maximum/y�
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_3/Maximum�
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_3/truediv�
&normalization_4/Reshape/ReadVariableOpReadVariableOp/normalization_4_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_4/Reshape/ReadVariableOp�
normalization_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_4/Reshape/shape�
normalization_4/ReshapeReshape.normalization_4/Reshape/ReadVariableOp:value:0&normalization_4/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_4/Reshape�
(normalization_4/Reshape_1/ReadVariableOpReadVariableOp1normalization_4_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_4/Reshape_1/ReadVariableOp�
normalization_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_4/Reshape_1/shape�
normalization_4/Reshape_1Reshape0normalization_4/Reshape_1/ReadVariableOp:value:0(normalization_4/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_4/Reshape_1�
normalization_4/subSubtwi normalization_4/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_4/sub�
normalization_4/SqrtSqrt"normalization_4/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_4/Sqrt{
normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_4/Maximum/y�
normalization_4/MaximumMaximumnormalization_4/Sqrt:y:0"normalization_4/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_4/Maximum�
normalization_4/truedivRealDivnormalization_4/sub:z:0normalization_4/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_4/truediv�
&normalization_5/Reshape/ReadVariableOpReadVariableOp/normalization_5_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_5/Reshape/ReadVariableOp�
normalization_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_5/Reshape/shape�
normalization_5/ReshapeReshape.normalization_5/Reshape/ReadVariableOp:value:0&normalization_5/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_5/Reshape�
(normalization_5/Reshape_1/ReadVariableOpReadVariableOp1normalization_5_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_5/Reshape_1/ReadVariableOp�
normalization_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_5/Reshape_1/shape�
normalization_5/Reshape_1Reshape0normalization_5/Reshape_1/ReadVariableOp:value:0(normalization_5/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_5/Reshape_1�
normalization_5/subSubcurv normalization_5/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_5/sub�
normalization_5/SqrtSqrt"normalization_5/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_5/Sqrt{
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_5/Maximum/y�
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_5/Maximum�
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_5/truediv�
&normalization_6/Reshape/ReadVariableOpReadVariableOp/normalization_6_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_6/Reshape/ReadVariableOp�
normalization_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_6/Reshape/shape�
normalization_6/ReshapeReshape.normalization_6/Reshape/ReadVariableOp:value:0&normalization_6/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_6/Reshape�
(normalization_6/Reshape_1/ReadVariableOpReadVariableOp1normalization_6_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_6/Reshape_1/ReadVariableOp�
normalization_6/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_6/Reshape_1/shape�
normalization_6/Reshape_1Reshape0normalization_6/Reshape_1/ReadVariableOp:value:0(normalization_6/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_6/Reshape_1�
normalization_6/subSublito normalization_6/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_6/sub�
normalization_6/SqrtSqrt"normalization_6/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_6/Sqrt{
normalization_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_6/Maximum/y�
normalization_6/MaximumMaximumnormalization_6/Sqrt:y:0"normalization_6/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_6/Maximum�
normalization_6/truedivRealDivnormalization_6/sub:z:0normalization_6/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_6/truediv�
&normalization_7/Reshape/ReadVariableOpReadVariableOp/normalization_7_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_7/Reshape/ReadVariableOp�
normalization_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_7/Reshape/shape�
normalization_7/ReshapeReshape.normalization_7/Reshape/ReadVariableOp:value:0&normalization_7/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_7/Reshape�
(normalization_7/Reshape_1/ReadVariableOpReadVariableOp1normalization_7_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_7/Reshape_1/ReadVariableOp�
normalization_7/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_7/Reshape_1/shape�
normalization_7/Reshape_1Reshape0normalization_7/Reshape_1/ReadVariableOp:value:0(normalization_7/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_7/Reshape_1�
normalization_7/subSubuso_solo normalization_7/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_7/sub�
normalization_7/SqrtSqrt"normalization_7/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_7/Sqrt{
normalization_7/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_7/Maximum/y�
normalization_7/MaximumMaximumnormalization_7/Sqrt:y:0"normalization_7/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_7/Maximum�
normalization_7/truedivRealDivnormalization_7/sub:z:0normalization_7/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_7/truediv�
concatenate/PartitionedCallPartitionedCallnormalization_1/truediv:z:0normalization_2/truediv:z:0normalization_3/truediv:z:0normalization_4/truediv:z:0normalization_5/truediv:z:0normalization_6/truediv:z:0normalization_7/truediv:z:0*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_3193092
concatenate/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_1_319914dense_1_319916*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3193222!
dense_1/StatefulPartitionedCall�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_3194212#
!dropout_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_319920dense_2_319922*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3193452!
dense_2/StatefulPartitionedCall�
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall'^normalization_1/Reshape/ReadVariableOp)^normalization_1/Reshape_1/ReadVariableOp'^normalization_2/Reshape/ReadVariableOp)^normalization_2/Reshape_1/ReadVariableOp'^normalization_3/Reshape/ReadVariableOp)^normalization_3/Reshape_1/ReadVariableOp'^normalization_4/Reshape/ReadVariableOp)^normalization_4/Reshape_1/ReadVariableOp'^normalization_5/Reshape/ReadVariableOp)^normalization_5/Reshape_1/ReadVariableOp'^normalization_6/Reshape/ReadVariableOp)^normalization_6/Reshape_1/ReadVariableOp'^normalization_7/Reshape/ReadVariableOp)^normalization_7/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������: : : : : : : : : : : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2P
&normalization_1/Reshape/ReadVariableOp&normalization_1/Reshape/ReadVariableOp2T
(normalization_1/Reshape_1/ReadVariableOp(normalization_1/Reshape_1/ReadVariableOp2P
&normalization_2/Reshape/ReadVariableOp&normalization_2/Reshape/ReadVariableOp2T
(normalization_2/Reshape_1/ReadVariableOp(normalization_2/Reshape_1/ReadVariableOp2P
&normalization_3/Reshape/ReadVariableOp&normalization_3/Reshape/ReadVariableOp2T
(normalization_3/Reshape_1/ReadVariableOp(normalization_3/Reshape_1/ReadVariableOp2P
&normalization_4/Reshape/ReadVariableOp&normalization_4/Reshape/ReadVariableOp2T
(normalization_4/Reshape_1/ReadVariableOp(normalization_4/Reshape_1/ReadVariableOp2P
&normalization_5/Reshape/ReadVariableOp&normalization_5/Reshape/ReadVariableOp2T
(normalization_5/Reshape_1/ReadVariableOp(normalization_5/Reshape_1/ReadVariableOp2P
&normalization_6/Reshape/ReadVariableOp&normalization_6/Reshape/ReadVariableOp2T
(normalization_6/Reshape_1/ReadVariableOp(normalization_6/Reshape_1/ReadVariableOp2P
&normalization_7/Reshape/ReadVariableOp&normalization_7/Reshape/ReadVariableOp2T
(normalization_7/Reshape_1/ReadVariableOp(normalization_7/Reshape_1/ReadVariableOp:N J
'
_output_shapes
:���������

_user_specified_nameslope:OK
'
_output_shapes
:���������
 
_user_specified_nameaspect:RN
'
_output_shapes
:���������
#
_user_specified_name	elevation:LH
'
_output_shapes
:���������

_user_specified_nametwi:MI
'
_output_shapes
:���������

_user_specified_namecurv:MI
'
_output_shapes
:���������

_user_specified_namelito:QM
'
_output_shapes
:���������
"
_user_specified_name
uso_solo
�.
�
__inference_adapt_step_320610
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�IteratorGetNext�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�add/ReadVariableOp�
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:���������*"
output_shapes
:���������*
output_types
2	2
IteratorGetNextb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsIteratorGetNext:components:0ExpandDims/dim:output:0*
T0	*'
_output_shapes
:���������2

ExpandDimsj
CastCastExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:���������2
Cast�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance}
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1V
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	2
Shapeu
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB"       2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis�
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:2

GatherV2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstX
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: 2
Prod|
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	2
add/ReadVariableOp_
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: 2
addW
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_1Q
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_2V
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: 2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/xO
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: 2
subp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpS
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: 2
mul]
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: 2
mul_1L
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: 2
add_1t
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1[
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: 2
sub_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yM
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: 2
powv
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2[
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: 2
add_2J
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: 2
mul_2[
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: 2
sub_2W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yS
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1_
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: 2
add_3N
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: 2
mul_3N
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: 2
add_4�
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignVariableOp�
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype02
AssignVariableOp_1�
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOp_2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
�
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_320729

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *HO�?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *�u�>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_model_layer_call_fn_319391	
slope

aspect
	elevation
twi
curv
lito
uso_solo
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: 

unknown_13:

unknown_14:

unknown_15:

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallslopeaspect	elevationtwicurvlitouso_solounknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_3193522
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:���������

_user_specified_nameslope:OK
'
_output_shapes
:���������
 
_user_specified_nameaspect:RN
'
_output_shapes
:���������
#
_user_specified_name	elevation:LH
'
_output_shapes
:���������

_user_specified_nametwi:MI
'
_output_shapes
:���������

_user_specified_namecurv:MI
'
_output_shapes
:���������

_user_specified_namelito:QM
'
_output_shapes
:���������
"
_user_specified_name
uso_solo
ϕ
�
A__inference_model_layer_call_and_return_conditional_losses_319813	
slope

aspect
	elevation
twi
curv
lito
uso_solo9
/normalization_1_reshape_readvariableop_resource: ;
1normalization_1_reshape_1_readvariableop_resource: 9
/normalization_2_reshape_readvariableop_resource: ;
1normalization_2_reshape_1_readvariableop_resource: 9
/normalization_3_reshape_readvariableop_resource: ;
1normalization_3_reshape_1_readvariableop_resource: 9
/normalization_4_reshape_readvariableop_resource: ;
1normalization_4_reshape_1_readvariableop_resource: 9
/normalization_5_reshape_readvariableop_resource: ;
1normalization_5_reshape_1_readvariableop_resource: 9
/normalization_6_reshape_readvariableop_resource: ;
1normalization_6_reshape_1_readvariableop_resource: 9
/normalization_7_reshape_readvariableop_resource: ;
1normalization_7_reshape_1_readvariableop_resource:  
dense_1_319801:
dense_1_319803: 
dense_2_319807:
dense_2_319809:
identity��dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�&normalization_1/Reshape/ReadVariableOp�(normalization_1/Reshape_1/ReadVariableOp�&normalization_2/Reshape/ReadVariableOp�(normalization_2/Reshape_1/ReadVariableOp�&normalization_3/Reshape/ReadVariableOp�(normalization_3/Reshape_1/ReadVariableOp�&normalization_4/Reshape/ReadVariableOp�(normalization_4/Reshape_1/ReadVariableOp�&normalization_5/Reshape/ReadVariableOp�(normalization_5/Reshape_1/ReadVariableOp�&normalization_6/Reshape/ReadVariableOp�(normalization_6/Reshape_1/ReadVariableOp�&normalization_7/Reshape/ReadVariableOp�(normalization_7/Reshape_1/ReadVariableOp�
&normalization_1/Reshape/ReadVariableOpReadVariableOp/normalization_1_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_1/Reshape/ReadVariableOp�
normalization_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_1/Reshape/shape�
normalization_1/ReshapeReshape.normalization_1/Reshape/ReadVariableOp:value:0&normalization_1/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_1/Reshape�
(normalization_1/Reshape_1/ReadVariableOpReadVariableOp1normalization_1_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_1/Reshape_1/ReadVariableOp�
normalization_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_1/Reshape_1/shape�
normalization_1/Reshape_1Reshape0normalization_1/Reshape_1/ReadVariableOp:value:0(normalization_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_1/Reshape_1�
normalization_1/subSubslope normalization_1/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_1/sub�
normalization_1/SqrtSqrt"normalization_1/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_1/Sqrt{
normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_1/Maximum/y�
normalization_1/MaximumMaximumnormalization_1/Sqrt:y:0"normalization_1/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_1/Maximum�
normalization_1/truedivRealDivnormalization_1/sub:z:0normalization_1/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_1/truediv�
&normalization_2/Reshape/ReadVariableOpReadVariableOp/normalization_2_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_2/Reshape/ReadVariableOp�
normalization_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_2/Reshape/shape�
normalization_2/ReshapeReshape.normalization_2/Reshape/ReadVariableOp:value:0&normalization_2/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_2/Reshape�
(normalization_2/Reshape_1/ReadVariableOpReadVariableOp1normalization_2_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_2/Reshape_1/ReadVariableOp�
normalization_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_2/Reshape_1/shape�
normalization_2/Reshape_1Reshape0normalization_2/Reshape_1/ReadVariableOp:value:0(normalization_2/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_2/Reshape_1�
normalization_2/subSubaspect normalization_2/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_2/sub�
normalization_2/SqrtSqrt"normalization_2/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_2/Sqrt{
normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_2/Maximum/y�
normalization_2/MaximumMaximumnormalization_2/Sqrt:y:0"normalization_2/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_2/Maximum�
normalization_2/truedivRealDivnormalization_2/sub:z:0normalization_2/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_2/truediv�
&normalization_3/Reshape/ReadVariableOpReadVariableOp/normalization_3_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_3/Reshape/ReadVariableOp�
normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_3/Reshape/shape�
normalization_3/ReshapeReshape.normalization_3/Reshape/ReadVariableOp:value:0&normalization_3/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_3/Reshape�
(normalization_3/Reshape_1/ReadVariableOpReadVariableOp1normalization_3_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_3/Reshape_1/ReadVariableOp�
normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_3/Reshape_1/shape�
normalization_3/Reshape_1Reshape0normalization_3/Reshape_1/ReadVariableOp:value:0(normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_3/Reshape_1�
normalization_3/subSub	elevation normalization_3/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_3/sub�
normalization_3/SqrtSqrt"normalization_3/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_3/Maximum/y�
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_3/Maximum�
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_3/truediv�
&normalization_4/Reshape/ReadVariableOpReadVariableOp/normalization_4_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_4/Reshape/ReadVariableOp�
normalization_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_4/Reshape/shape�
normalization_4/ReshapeReshape.normalization_4/Reshape/ReadVariableOp:value:0&normalization_4/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_4/Reshape�
(normalization_4/Reshape_1/ReadVariableOpReadVariableOp1normalization_4_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_4/Reshape_1/ReadVariableOp�
normalization_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_4/Reshape_1/shape�
normalization_4/Reshape_1Reshape0normalization_4/Reshape_1/ReadVariableOp:value:0(normalization_4/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_4/Reshape_1�
normalization_4/subSubtwi normalization_4/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_4/sub�
normalization_4/SqrtSqrt"normalization_4/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_4/Sqrt{
normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_4/Maximum/y�
normalization_4/MaximumMaximumnormalization_4/Sqrt:y:0"normalization_4/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_4/Maximum�
normalization_4/truedivRealDivnormalization_4/sub:z:0normalization_4/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_4/truediv�
&normalization_5/Reshape/ReadVariableOpReadVariableOp/normalization_5_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_5/Reshape/ReadVariableOp�
normalization_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_5/Reshape/shape�
normalization_5/ReshapeReshape.normalization_5/Reshape/ReadVariableOp:value:0&normalization_5/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_5/Reshape�
(normalization_5/Reshape_1/ReadVariableOpReadVariableOp1normalization_5_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_5/Reshape_1/ReadVariableOp�
normalization_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_5/Reshape_1/shape�
normalization_5/Reshape_1Reshape0normalization_5/Reshape_1/ReadVariableOp:value:0(normalization_5/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_5/Reshape_1�
normalization_5/subSubcurv normalization_5/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_5/sub�
normalization_5/SqrtSqrt"normalization_5/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_5/Sqrt{
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_5/Maximum/y�
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_5/Maximum�
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_5/truediv�
&normalization_6/Reshape/ReadVariableOpReadVariableOp/normalization_6_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_6/Reshape/ReadVariableOp�
normalization_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_6/Reshape/shape�
normalization_6/ReshapeReshape.normalization_6/Reshape/ReadVariableOp:value:0&normalization_6/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_6/Reshape�
(normalization_6/Reshape_1/ReadVariableOpReadVariableOp1normalization_6_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_6/Reshape_1/ReadVariableOp�
normalization_6/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_6/Reshape_1/shape�
normalization_6/Reshape_1Reshape0normalization_6/Reshape_1/ReadVariableOp:value:0(normalization_6/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_6/Reshape_1�
normalization_6/subSublito normalization_6/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_6/sub�
normalization_6/SqrtSqrt"normalization_6/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_6/Sqrt{
normalization_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_6/Maximum/y�
normalization_6/MaximumMaximumnormalization_6/Sqrt:y:0"normalization_6/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_6/Maximum�
normalization_6/truedivRealDivnormalization_6/sub:z:0normalization_6/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_6/truediv�
&normalization_7/Reshape/ReadVariableOpReadVariableOp/normalization_7_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_7/Reshape/ReadVariableOp�
normalization_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_7/Reshape/shape�
normalization_7/ReshapeReshape.normalization_7/Reshape/ReadVariableOp:value:0&normalization_7/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_7/Reshape�
(normalization_7/Reshape_1/ReadVariableOpReadVariableOp1normalization_7_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_7/Reshape_1/ReadVariableOp�
normalization_7/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_7/Reshape_1/shape�
normalization_7/Reshape_1Reshape0normalization_7/Reshape_1/ReadVariableOp:value:0(normalization_7/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_7/Reshape_1�
normalization_7/subSubuso_solo normalization_7/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_7/sub�
normalization_7/SqrtSqrt"normalization_7/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_7/Sqrt{
normalization_7/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_7/Maximum/y�
normalization_7/MaximumMaximumnormalization_7/Sqrt:y:0"normalization_7/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_7/Maximum�
normalization_7/truedivRealDivnormalization_7/sub:z:0normalization_7/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_7/truediv�
concatenate/PartitionedCallPartitionedCallnormalization_1/truediv:z:0normalization_2/truediv:z:0normalization_3/truediv:z:0normalization_4/truediv:z:0normalization_5/truediv:z:0normalization_6/truediv:z:0normalization_7/truediv:z:0*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_3193092
concatenate/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_1_319801dense_1_319803*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3193222!
dense_1/StatefulPartitionedCall�
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_3193332
dropout_1/PartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_319807dense_2_319809*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3193452!
dense_2/StatefulPartitionedCall�
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall'^normalization_1/Reshape/ReadVariableOp)^normalization_1/Reshape_1/ReadVariableOp'^normalization_2/Reshape/ReadVariableOp)^normalization_2/Reshape_1/ReadVariableOp'^normalization_3/Reshape/ReadVariableOp)^normalization_3/Reshape_1/ReadVariableOp'^normalization_4/Reshape/ReadVariableOp)^normalization_4/Reshape_1/ReadVariableOp'^normalization_5/Reshape/ReadVariableOp)^normalization_5/Reshape_1/ReadVariableOp'^normalization_6/Reshape/ReadVariableOp)^normalization_6/Reshape_1/ReadVariableOp'^normalization_7/Reshape/ReadVariableOp)^normalization_7/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������: : : : : : : : : : : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2P
&normalization_1/Reshape/ReadVariableOp&normalization_1/Reshape/ReadVariableOp2T
(normalization_1/Reshape_1/ReadVariableOp(normalization_1/Reshape_1/ReadVariableOp2P
&normalization_2/Reshape/ReadVariableOp&normalization_2/Reshape/ReadVariableOp2T
(normalization_2/Reshape_1/ReadVariableOp(normalization_2/Reshape_1/ReadVariableOp2P
&normalization_3/Reshape/ReadVariableOp&normalization_3/Reshape/ReadVariableOp2T
(normalization_3/Reshape_1/ReadVariableOp(normalization_3/Reshape_1/ReadVariableOp2P
&normalization_4/Reshape/ReadVariableOp&normalization_4/Reshape/ReadVariableOp2T
(normalization_4/Reshape_1/ReadVariableOp(normalization_4/Reshape_1/ReadVariableOp2P
&normalization_5/Reshape/ReadVariableOp&normalization_5/Reshape/ReadVariableOp2T
(normalization_5/Reshape_1/ReadVariableOp(normalization_5/Reshape_1/ReadVariableOp2P
&normalization_6/Reshape/ReadVariableOp&normalization_6/Reshape/ReadVariableOp2T
(normalization_6/Reshape_1/ReadVariableOp(normalization_6/Reshape_1/ReadVariableOp2P
&normalization_7/Reshape/ReadVariableOp&normalization_7/Reshape/ReadVariableOp2T
(normalization_7/Reshape_1/ReadVariableOp(normalization_7/Reshape_1/ReadVariableOp:N J
'
_output_shapes
:���������

_user_specified_nameslope:OK
'
_output_shapes
:���������
 
_user_specified_nameaspect:RN
'
_output_shapes
:���������
#
_user_specified_name	elevation:LH
'
_output_shapes
:���������

_user_specified_nametwi:MI
'
_output_shapes
:���������

_user_specified_namecurv:MI
'
_output_shapes
:���������

_user_specified_namelito:QM
'
_output_shapes
:���������
"
_user_specified_name
uso_solo
�.
�
__inference_adapt_step_320414
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�IteratorGetNext�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�add/ReadVariableOp�
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:���������*"
output_shapes
:���������*
output_types
22
IteratorGetNextb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsIteratorGetNext:components:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������2

ExpandDimsj
CastCastExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance}
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1V
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	2
Shapeu
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB"       2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis�
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:2

GatherV2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstX
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: 2
Prod|
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	2
add/ReadVariableOp_
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: 2
addW
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_1Q
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_2V
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: 2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/xO
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: 2
subp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpS
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: 2
mul]
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: 2
mul_1L
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: 2
add_1t
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1[
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: 2
sub_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yM
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: 2
powv
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2[
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: 2
add_2J
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: 2
mul_2[
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: 2
sub_2W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yS
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1_
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: 2
add_3N
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: 2
mul_3N
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: 2
add_4�
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignVariableOp�
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype02
AssignVariableOp_1�
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOp_2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
�.
�
__inference_adapt_step_320365
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�IteratorGetNext�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�add/ReadVariableOp�
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:���������*"
output_shapes
:���������*
output_types
22
IteratorGetNextb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsIteratorGetNext:components:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������2

ExpandDimsj
CastCastExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance}
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1V
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	2
Shapeu
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB"       2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis�
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:2

GatherV2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstX
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: 2
Prod|
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	2
add/ReadVariableOp_
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: 2
addW
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_1Q
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_2V
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: 2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/xO
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: 2
subp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpS
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: 2
mul]
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: 2
mul_1L
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: 2
add_1t
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1[
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: 2
sub_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yM
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: 2
powv
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2[
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: 2
add_2J
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: 2
mul_2[
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: 2
sub_2W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yS
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1_
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: 2
add_3N
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: 2
mul_3N
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: 2
add_4�
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignVariableOp�
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype02
AssignVariableOp_1�
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOp_2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
֣
�
A__inference_model_layer_call_and_return_conditional_losses_320316
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_69
/normalization_1_reshape_readvariableop_resource: ;
1normalization_1_reshape_1_readvariableop_resource: 9
/normalization_2_reshape_readvariableop_resource: ;
1normalization_2_reshape_1_readvariableop_resource: 9
/normalization_3_reshape_readvariableop_resource: ;
1normalization_3_reshape_1_readvariableop_resource: 9
/normalization_4_reshape_readvariableop_resource: ;
1normalization_4_reshape_1_readvariableop_resource: 9
/normalization_5_reshape_readvariableop_resource: ;
1normalization_5_reshape_1_readvariableop_resource: 9
/normalization_6_reshape_readvariableop_resource: ;
1normalization_6_reshape_1_readvariableop_resource: 9
/normalization_7_reshape_readvariableop_resource: ;
1normalization_7_reshape_1_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identity��dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�&normalization_1/Reshape/ReadVariableOp�(normalization_1/Reshape_1/ReadVariableOp�&normalization_2/Reshape/ReadVariableOp�(normalization_2/Reshape_1/ReadVariableOp�&normalization_3/Reshape/ReadVariableOp�(normalization_3/Reshape_1/ReadVariableOp�&normalization_4/Reshape/ReadVariableOp�(normalization_4/Reshape_1/ReadVariableOp�&normalization_5/Reshape/ReadVariableOp�(normalization_5/Reshape_1/ReadVariableOp�&normalization_6/Reshape/ReadVariableOp�(normalization_6/Reshape_1/ReadVariableOp�&normalization_7/Reshape/ReadVariableOp�(normalization_7/Reshape_1/ReadVariableOp�
&normalization_1/Reshape/ReadVariableOpReadVariableOp/normalization_1_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_1/Reshape/ReadVariableOp�
normalization_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_1/Reshape/shape�
normalization_1/ReshapeReshape.normalization_1/Reshape/ReadVariableOp:value:0&normalization_1/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_1/Reshape�
(normalization_1/Reshape_1/ReadVariableOpReadVariableOp1normalization_1_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_1/Reshape_1/ReadVariableOp�
normalization_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_1/Reshape_1/shape�
normalization_1/Reshape_1Reshape0normalization_1/Reshape_1/ReadVariableOp:value:0(normalization_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_1/Reshape_1�
normalization_1/subSubinputs_0 normalization_1/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_1/sub�
normalization_1/SqrtSqrt"normalization_1/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_1/Sqrt{
normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_1/Maximum/y�
normalization_1/MaximumMaximumnormalization_1/Sqrt:y:0"normalization_1/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_1/Maximum�
normalization_1/truedivRealDivnormalization_1/sub:z:0normalization_1/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_1/truediv�
&normalization_2/Reshape/ReadVariableOpReadVariableOp/normalization_2_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_2/Reshape/ReadVariableOp�
normalization_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_2/Reshape/shape�
normalization_2/ReshapeReshape.normalization_2/Reshape/ReadVariableOp:value:0&normalization_2/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_2/Reshape�
(normalization_2/Reshape_1/ReadVariableOpReadVariableOp1normalization_2_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_2/Reshape_1/ReadVariableOp�
normalization_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_2/Reshape_1/shape�
normalization_2/Reshape_1Reshape0normalization_2/Reshape_1/ReadVariableOp:value:0(normalization_2/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_2/Reshape_1�
normalization_2/subSubinputs_1 normalization_2/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_2/sub�
normalization_2/SqrtSqrt"normalization_2/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_2/Sqrt{
normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_2/Maximum/y�
normalization_2/MaximumMaximumnormalization_2/Sqrt:y:0"normalization_2/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_2/Maximum�
normalization_2/truedivRealDivnormalization_2/sub:z:0normalization_2/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_2/truediv�
&normalization_3/Reshape/ReadVariableOpReadVariableOp/normalization_3_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_3/Reshape/ReadVariableOp�
normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_3/Reshape/shape�
normalization_3/ReshapeReshape.normalization_3/Reshape/ReadVariableOp:value:0&normalization_3/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_3/Reshape�
(normalization_3/Reshape_1/ReadVariableOpReadVariableOp1normalization_3_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_3/Reshape_1/ReadVariableOp�
normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_3/Reshape_1/shape�
normalization_3/Reshape_1Reshape0normalization_3/Reshape_1/ReadVariableOp:value:0(normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_3/Reshape_1�
normalization_3/subSubinputs_2 normalization_3/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_3/sub�
normalization_3/SqrtSqrt"normalization_3/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_3/Maximum/y�
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_3/Maximum�
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_3/truediv�
&normalization_4/Reshape/ReadVariableOpReadVariableOp/normalization_4_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_4/Reshape/ReadVariableOp�
normalization_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_4/Reshape/shape�
normalization_4/ReshapeReshape.normalization_4/Reshape/ReadVariableOp:value:0&normalization_4/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_4/Reshape�
(normalization_4/Reshape_1/ReadVariableOpReadVariableOp1normalization_4_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_4/Reshape_1/ReadVariableOp�
normalization_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_4/Reshape_1/shape�
normalization_4/Reshape_1Reshape0normalization_4/Reshape_1/ReadVariableOp:value:0(normalization_4/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_4/Reshape_1�
normalization_4/subSubinputs_3 normalization_4/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_4/sub�
normalization_4/SqrtSqrt"normalization_4/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_4/Sqrt{
normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_4/Maximum/y�
normalization_4/MaximumMaximumnormalization_4/Sqrt:y:0"normalization_4/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_4/Maximum�
normalization_4/truedivRealDivnormalization_4/sub:z:0normalization_4/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_4/truediv�
&normalization_5/Reshape/ReadVariableOpReadVariableOp/normalization_5_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_5/Reshape/ReadVariableOp�
normalization_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_5/Reshape/shape�
normalization_5/ReshapeReshape.normalization_5/Reshape/ReadVariableOp:value:0&normalization_5/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_5/Reshape�
(normalization_5/Reshape_1/ReadVariableOpReadVariableOp1normalization_5_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_5/Reshape_1/ReadVariableOp�
normalization_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_5/Reshape_1/shape�
normalization_5/Reshape_1Reshape0normalization_5/Reshape_1/ReadVariableOp:value:0(normalization_5/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_5/Reshape_1�
normalization_5/subSubinputs_4 normalization_5/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_5/sub�
normalization_5/SqrtSqrt"normalization_5/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_5/Sqrt{
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_5/Maximum/y�
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_5/Maximum�
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_5/truediv�
&normalization_6/Reshape/ReadVariableOpReadVariableOp/normalization_6_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_6/Reshape/ReadVariableOp�
normalization_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_6/Reshape/shape�
normalization_6/ReshapeReshape.normalization_6/Reshape/ReadVariableOp:value:0&normalization_6/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_6/Reshape�
(normalization_6/Reshape_1/ReadVariableOpReadVariableOp1normalization_6_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_6/Reshape_1/ReadVariableOp�
normalization_6/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_6/Reshape_1/shape�
normalization_6/Reshape_1Reshape0normalization_6/Reshape_1/ReadVariableOp:value:0(normalization_6/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_6/Reshape_1�
normalization_6/subSubinputs_5 normalization_6/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_6/sub�
normalization_6/SqrtSqrt"normalization_6/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_6/Sqrt{
normalization_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_6/Maximum/y�
normalization_6/MaximumMaximumnormalization_6/Sqrt:y:0"normalization_6/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_6/Maximum�
normalization_6/truedivRealDivnormalization_6/sub:z:0normalization_6/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_6/truediv�
&normalization_7/Reshape/ReadVariableOpReadVariableOp/normalization_7_reshape_readvariableop_resource*
_output_shapes
: *
dtype02(
&normalization_7/Reshape/ReadVariableOp�
normalization_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_7/Reshape/shape�
normalization_7/ReshapeReshape.normalization_7/Reshape/ReadVariableOp:value:0&normalization_7/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_7/Reshape�
(normalization_7/Reshape_1/ReadVariableOpReadVariableOp1normalization_7_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalization_7/Reshape_1/ReadVariableOp�
normalization_7/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_7/Reshape_1/shape�
normalization_7/Reshape_1Reshape0normalization_7/Reshape_1/ReadVariableOp:value:0(normalization_7/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_7/Reshape_1�
normalization_7/subSubinputs_6 normalization_7/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_7/sub�
normalization_7/SqrtSqrt"normalization_7/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_7/Sqrt{
normalization_7/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization_7/Maximum/y�
normalization_7/MaximumMaximumnormalization_7/Sqrt:y:0"normalization_7/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_7/Maximum�
normalization_7/truedivRealDivnormalization_7/sub:z:0normalization_7/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization_7/truedivt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis�
concatenate/concatConcatV2normalization_1/truediv:z:0normalization_2/truediv:z:0normalization_3/truediv:z:0normalization_4/truediv:z:0normalization_5/truediv:z:0normalization_6/truediv:z:0normalization_7/truediv:z:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatenate/concat�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulconcatenate/concat:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/BiasAddp
dense_1/TanhTanhdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_1/Tanhw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *HO�?2
dropout_1/dropout/Const�
dropout_1/dropout/MulMuldense_1/Tanh:y:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:���������2
dropout_1/dropout/Mulr
dropout_1/dropout/ShapeShapedense_1/Tanh:y:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape�
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform�
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *�u�>2"
 dropout_1/dropout/GreaterEqual/y�
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������2 
dropout_1/dropout/GreaterEqual�
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������2
dropout_1/dropout/Cast�
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:���������2
dropout_1/dropout/Mul_1�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_2/BiasAdd�
IdentityIdentitydense_2/BiasAdd:output:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp'^normalization_1/Reshape/ReadVariableOp)^normalization_1/Reshape_1/ReadVariableOp'^normalization_2/Reshape/ReadVariableOp)^normalization_2/Reshape_1/ReadVariableOp'^normalization_3/Reshape/ReadVariableOp)^normalization_3/Reshape_1/ReadVariableOp'^normalization_4/Reshape/ReadVariableOp)^normalization_4/Reshape_1/ReadVariableOp'^normalization_5/Reshape/ReadVariableOp)^normalization_5/Reshape_1/ReadVariableOp'^normalization_6/Reshape/ReadVariableOp)^normalization_6/Reshape_1/ReadVariableOp'^normalization_7/Reshape/ReadVariableOp)^normalization_7/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������: : : : : : : : : : : : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2P
&normalization_1/Reshape/ReadVariableOp&normalization_1/Reshape/ReadVariableOp2T
(normalization_1/Reshape_1/ReadVariableOp(normalization_1/Reshape_1/ReadVariableOp2P
&normalization_2/Reshape/ReadVariableOp&normalization_2/Reshape/ReadVariableOp2T
(normalization_2/Reshape_1/ReadVariableOp(normalization_2/Reshape_1/ReadVariableOp2P
&normalization_3/Reshape/ReadVariableOp&normalization_3/Reshape/ReadVariableOp2T
(normalization_3/Reshape_1/ReadVariableOp(normalization_3/Reshape_1/ReadVariableOp2P
&normalization_4/Reshape/ReadVariableOp&normalization_4/Reshape/ReadVariableOp2T
(normalization_4/Reshape_1/ReadVariableOp(normalization_4/Reshape_1/ReadVariableOp2P
&normalization_5/Reshape/ReadVariableOp&normalization_5/Reshape/ReadVariableOp2T
(normalization_5/Reshape_1/ReadVariableOp(normalization_5/Reshape_1/ReadVariableOp2P
&normalization_6/Reshape/ReadVariableOp&normalization_6/Reshape/ReadVariableOp2T
(normalization_6/Reshape_1/ReadVariableOp(normalization_6/Reshape_1/ReadVariableOp2P
&normalization_7/Reshape/ReadVariableOp&normalization_7/Reshape/ReadVariableOp2T
(normalization_7/Reshape_1/ReadVariableOp(normalization_7/Reshape_1/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/6
�
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_319421

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *HO�?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *�u�>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
9
aspect/
serving_default_aspect:0���������
5
curv-
serving_default_curv:0���������
?
	elevation2
serving_default_elevation:0���������
5
lito-
serving_default_lito:0���������
7
slope.
serving_default_slope:0���������
3
twi,
serving_default_twi:0���������
=
uso_solo1
serving_default_uso_solo:0���������;
dense_20
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�z
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-0
layer-7
	layer_with_weights-1
	layer-8

layer_with_weights-2

layer-9
layer_with_weights-3
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer-16
layer_with_weights-8
layer-17
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"�u
_tf_keras_network�u{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "slope"}, "name": "slope", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "aspect"}, "name": "aspect", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "elevation"}, "name": "elevation", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "twi"}, "name": "twi", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "curv"}, "name": "curv", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lito"}, "name": "lito", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "uso_solo"}, "name": "uso_solo", "inbound_nodes": []}, {"class_name": "Normalization", "config": {"name": "normalization_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_1", "inbound_nodes": [[["slope", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_2", "inbound_nodes": [[["aspect", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_3", "inbound_nodes": [[["elevation", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_4", "inbound_nodes": [[["twi", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_5", "inbound_nodes": [[["curv", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_6", "inbound_nodes": [[["lito", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_7", "inbound_nodes": [[["uso_solo", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["normalization_1", 0, 0, {}], ["normalization_2", 0, 0, {}], ["normalization_3", 0, 0, {}], ["normalization_4", 0, 0, {}], ["normalization_5", 0, 0, {}], ["normalization_6", 0, 0, {}], ["normalization_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 12, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2821478566400208, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}], "input_layers": [["slope", 0, 0], ["aspect", 0, 0], ["elevation", 0, 0], ["twi", 0, 0], ["curv", 0, 0], ["lito", 0, 0], ["uso_solo", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "shared_object_id": 22, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "slope"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "aspect"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "elevation"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "twi"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "curv"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "lito"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "uso_solo"]}], "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "slope"}, "name": "slope", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "aspect"}, "name": "aspect", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "elevation"}, "name": "elevation", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "twi"}, "name": "twi", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "curv"}, "name": "curv", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lito"}, "name": "lito", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "uso_solo"}, "name": "uso_solo", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "Normalization", "config": {"name": "normalization_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_1", "inbound_nodes": [[["slope", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Normalization", "config": {"name": "normalization_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_2", "inbound_nodes": [[["aspect", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "Normalization", "config": {"name": "normalization_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_3", "inbound_nodes": [[["elevation", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Normalization", "config": {"name": "normalization_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_4", "inbound_nodes": [[["twi", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Normalization", "config": {"name": "normalization_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_5", "inbound_nodes": [[["curv", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Normalization", "config": {"name": "normalization_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_6", "inbound_nodes": [[["lito", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Normalization", "config": {"name": "normalization_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_7", "inbound_nodes": [[["uso_solo", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["normalization_1", 0, 0, {}], ["normalization_2", 0, 0, {}], ["normalization_3", 0, 0, {}], ["normalization_4", 0, 0, {}], ["normalization_5", 0, 0, {}], ["normalization_6", 0, 0, {}], ["normalization_7", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 12, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["concatenate", 0, 0, {}]]], "shared_object_id": 17}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2821478566400208, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dropout_1", 0, 0, {}]]], "shared_object_id": 21}], "input_layers": [["slope", 0, 0], ["aspect", 0, 0], ["elevation", 0, 0], ["twi", 0, 0], ["curv", 0, 0], ["lito", 0, 0], ["uso_solo", 0, 0]], "output_layers": [["dense_2", 0, 0]]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": true, "label_smoothing": 0}, "shared_object_id": 30}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 31}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "slope", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "slope"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "aspect", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "aspect"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "elevation", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "elevation"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "twi", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "twi"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "curv", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "curv"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "lito", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lito"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "uso_solo", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "uso_solo"}}
�

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean
variance
	count
 	keras_api
�_adapt_function"�
_tf_keras_layer�{"name": "normalization_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "stateful": true, "must_restore_from_config": true, "class_name": "Normalization", "config": {"name": "normalization_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "inbound_nodes": [[["slope", 0, 0, {}]]], "shared_object_id": 7, "build_input_shape": [null, 1]}
�
!
_keep_axis
"_reduce_axis
#_reduce_axis_mask
$_broadcast_shape
%mean
&variance
	'count
(	keras_api
�_adapt_function"�
_tf_keras_layer�{"name": "normalization_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "stateful": true, "must_restore_from_config": true, "class_name": "Normalization", "config": {"name": "normalization_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "inbound_nodes": [[["aspect", 0, 0, {}]]], "shared_object_id": 8, "build_input_shape": [null, 1]}
�
)
_keep_axis
*_reduce_axis
+_reduce_axis_mask
,_broadcast_shape
-mean
.variance
	/count
0	keras_api
�_adapt_function"�
_tf_keras_layer�{"name": "normalization_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "stateful": true, "must_restore_from_config": true, "class_name": "Normalization", "config": {"name": "normalization_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "inbound_nodes": [[["elevation", 0, 0, {}]]], "shared_object_id": 9, "build_input_shape": [null, 1]}
�
1
_keep_axis
2_reduce_axis
3_reduce_axis_mask
4_broadcast_shape
5mean
6variance
	7count
8	keras_api
�_adapt_function"�
_tf_keras_layer�{"name": "normalization_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "stateful": true, "must_restore_from_config": true, "class_name": "Normalization", "config": {"name": "normalization_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "inbound_nodes": [[["twi", 0, 0, {}]]], "shared_object_id": 10, "build_input_shape": [null, 1]}
�
9
_keep_axis
:_reduce_axis
;_reduce_axis_mask
<_broadcast_shape
=mean
>variance
	?count
@	keras_api
�_adapt_function"�
_tf_keras_layer�{"name": "normalization_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "stateful": true, "must_restore_from_config": true, "class_name": "Normalization", "config": {"name": "normalization_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "inbound_nodes": [[["curv", 0, 0, {}]]], "shared_object_id": 11, "build_input_shape": [null, 1]}
�
A
_keep_axis
B_reduce_axis
C_reduce_axis_mask
D_broadcast_shape
Emean
Fvariance
	Gcount
H	keras_api
�_adapt_function"�
_tf_keras_layer�{"name": "normalization_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "stateful": true, "must_restore_from_config": true, "class_name": "Normalization", "config": {"name": "normalization_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "inbound_nodes": [[["lito", 0, 0, {}]]], "shared_object_id": 12, "build_input_shape": [null, 1]}
�
I
_keep_axis
J_reduce_axis
K_reduce_axis_mask
L_broadcast_shape
Mmean
Nvariance
	Ocount
P	keras_api
�_adapt_function"�
_tf_keras_layer�{"name": "normalization_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "stateful": true, "must_restore_from_config": true, "class_name": "Normalization", "config": {"name": "normalization_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "inbound_nodes": [[["uso_solo", 0, 0, {}]]], "shared_object_id": 13, "build_input_shape": [null, 1]}
�
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["normalization_1", 0, 0, {}], ["normalization_2", 0, 0, {}], ["normalization_3", 0, 0, {}], ["normalization_4", 0, 0, {}], ["normalization_5", 0, 0, {}], ["normalization_6", 0, 0, {}], ["normalization_7", 0, 0, {}]]], "shared_object_id": 14, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}]}
�	

Ukernel
Vbias
Wtrainable_variables
Xregularization_losses
Y	variables
Z	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 12, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concatenate", 0, 0, {}]]], "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 7}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7]}}
�
[trainable_variables
\regularization_losses
]	variables
^	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2821478566400208, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 18}
�	

_kernel
`bias
atrainable_variables
bregularization_losses
c	variables
d	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_1", 0, 0, {}]]], "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 12}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12]}}
�
eiter

fbeta_1

gbeta_2
	hdecay
ilearning_rateUm�Vm�_m�`m�Uv�Vv�_v�`v�"
	optimizer
<
U0
V1
_2
`3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
0
1
2
%3
&4
'5
-6
.7
/8
59
610
711
=12
>13
?14
E15
F16
G17
M18
N19
O20
U21
V22
_23
`24"
trackable_list_wrapper
�
trainable_variables
jlayer_regularization_losses
knon_trainable_variables

llayers
mmetrics
regularization_losses
nlayer_metrics
	variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Qtrainable_variables
olayer_regularization_losses
pnon_trainable_variables

qlayers
rmetrics
Rregularization_losses
slayer_metrics
S	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :2dense_1/kernel
:2dense_1/bias
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
Wtrainable_variables
tlayer_regularization_losses
unon_trainable_variables

vlayers
wmetrics
Xregularization_losses
xlayer_metrics
Y	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
[trainable_variables
ylayer_regularization_losses
znon_trainable_variables

{layers
|metrics
\regularization_losses
}layer_metrics
]	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :2dense_2/kernel
:2dense_2/bias
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
�
atrainable_variables
~layer_regularization_losses
non_trainable_variables
�layers
�metrics
bregularization_losses
�layer_metrics
c	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
�
0
1
2
%3
&4
'5
-6
.7
/8
59
610
711
=12
>13
?14
E15
F16
G17
M18
N19
O20"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17"
trackable_list_wrapper
0
�0
�1"
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
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 34}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 31}
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
%:#2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
%:#2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
%:#2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
%:#2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
�2�
&__inference_model_layer_call_fn_319391
&__inference_model_layer_call_fn_320028
&__inference_model_layer_call_fn_320075
&__inference_model_layer_call_fn_319700�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
A__inference_model_layer_call_and_return_conditional_losses_320192
A__inference_model_layer_call_and_return_conditional_losses_320316
A__inference_model_layer_call_and_return_conditional_losses_319813
A__inference_model_layer_call_and_return_conditional_losses_319926�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
!__inference__wrapped_model_319187�
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
�
slope���������
 �
aspect���������
#� 
	elevation���������
�
twi���������
�
curv���������
�
lito���������
"�
uso_solo���������
�2�
__inference_adapt_step_320365�
���
FullArgSpec
args�

jiterator
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
__inference_adapt_step_320414�
���
FullArgSpec
args�

jiterator
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
__inference_adapt_step_320463�
���
FullArgSpec
args�

jiterator
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
__inference_adapt_step_320512�
���
FullArgSpec
args�

jiterator
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
__inference_adapt_step_320561�
���
FullArgSpec
args�

jiterator
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
__inference_adapt_step_320610�
���
FullArgSpec
args�

jiterator
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
__inference_adapt_step_320659�
���
FullArgSpec
args�

jiterator
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
,__inference_concatenate_layer_call_fn_320670�
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
�2�
G__inference_concatenate_layer_call_and_return_conditional_losses_320682�
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
�2�
(__inference_dense_1_layer_call_fn_320691�
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
�2�
C__inference_dense_1_layer_call_and_return_conditional_losses_320702�
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
�2�
*__inference_dropout_1_layer_call_fn_320707
*__inference_dropout_1_layer_call_fn_320712�
���
FullArgSpec)
args!�
jself
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
annotations� *
 
�2�
E__inference_dropout_1_layer_call_and_return_conditional_losses_320717
E__inference_dropout_1_layer_call_and_return_conditional_losses_320729�
���
FullArgSpec)
args!�
jself
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
annotations� *
 
�2�
(__inference_dense_2_layer_call_fn_320738�
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
�2�
C__inference_dense_2_layer_call_and_return_conditional_losses_320748�
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
$__inference_signature_wrapper_319981aspectcurv	elevationlitoslopetwiuso_solo"�
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
 �
!__inference__wrapped_model_319187�%&-.56=>EFMNUV_`���
���
���
�
slope���������
 �
aspect���������
#� 
	elevation���������
�
twi���������
�
curv���������
�
lito���������
"�
uso_solo���������
� "1�.
,
dense_2!�
dense_2���������i
__inference_adapt_step_320365H=�:
3�0
.�+�
����������IteratorSpec
� "
 i
__inference_adapt_step_320414H'%&=�:
3�0
.�+�
����������IteratorSpec
� "
 i
__inference_adapt_step_320463H/-.=�:
3�0
.�+�
����������IteratorSpec
� "
 i
__inference_adapt_step_320512H756=�:
3�0
.�+�
����������IteratorSpec
� "
 i
__inference_adapt_step_320561H?=>=�:
3�0
.�+�
����������IteratorSpec
� "
 i
__inference_adapt_step_320610HGEF=�:
3�0
.�+�
����������	IteratorSpec
� "
 i
__inference_adapt_step_320659HOMN=�:
3�0
.�+�
����������	IteratorSpec
� "
 �
G__inference_concatenate_layer_call_and_return_conditional_losses_320682����
���
���
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
"�
inputs/3���������
"�
inputs/4���������
"�
inputs/5���������
"�
inputs/6���������
� "%�"
�
0���������
� �
,__inference_concatenate_layer_call_fn_320670����
���
���
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
"�
inputs/3���������
"�
inputs/4���������
"�
inputs/5���������
"�
inputs/6���������
� "�����������
C__inference_dense_1_layer_call_and_return_conditional_losses_320702\UV/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� {
(__inference_dense_1_layer_call_fn_320691OUV/�,
%�"
 �
inputs���������
� "�����������
C__inference_dense_2_layer_call_and_return_conditional_losses_320748\_`/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� {
(__inference_dense_2_layer_call_fn_320738O_`/�,
%�"
 �
inputs���������
� "�����������
E__inference_dropout_1_layer_call_and_return_conditional_losses_320717\3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
E__inference_dropout_1_layer_call_and_return_conditional_losses_320729\3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� }
*__inference_dropout_1_layer_call_fn_320707O3�0
)�&
 �
inputs���������
p 
� "����������}
*__inference_dropout_1_layer_call_fn_320712O3�0
)�&
 �
inputs���������
p
� "�����������
A__inference_model_layer_call_and_return_conditional_losses_319813�%&-.56=>EFMNUV_`���
���
���
�
slope���������
 �
aspect���������
#� 
	elevation���������
�
twi���������
�
curv���������
�
lito���������
"�
uso_solo���������
p 

 
� "%�"
�
0���������
� �
A__inference_model_layer_call_and_return_conditional_losses_319926�%&-.56=>EFMNUV_`���
���
���
�
slope���������
 �
aspect���������
#� 
	elevation���������
�
twi���������
�
curv���������
�
lito���������
"�
uso_solo���������
p

 
� "%�"
�
0���������
� �
A__inference_model_layer_call_and_return_conditional_losses_320192�%&-.56=>EFMNUV_`���
���
���
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
"�
inputs/3���������
"�
inputs/4���������
"�
inputs/5���������
"�
inputs/6���������
p 

 
� "%�"
�
0���������
� �
A__inference_model_layer_call_and_return_conditional_losses_320316�%&-.56=>EFMNUV_`���
���
���
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
"�
inputs/3���������
"�
inputs/4���������
"�
inputs/5���������
"�
inputs/6���������
p

 
� "%�"
�
0���������
� �
&__inference_model_layer_call_fn_319391�%&-.56=>EFMNUV_`���
���
���
�
slope���������
 �
aspect���������
#� 
	elevation���������
�
twi���������
�
curv���������
�
lito���������
"�
uso_solo���������
p 

 
� "�����������
&__inference_model_layer_call_fn_319700�%&-.56=>EFMNUV_`���
���
���
�
slope���������
 �
aspect���������
#� 
	elevation���������
�
twi���������
�
curv���������
�
lito���������
"�
uso_solo���������
p

 
� "�����������
&__inference_model_layer_call_fn_320028�%&-.56=>EFMNUV_`���
���
���
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
"�
inputs/3���������
"�
inputs/4���������
"�
inputs/5���������
"�
inputs/6���������
p 

 
� "�����������
&__inference_model_layer_call_fn_320075�%&-.56=>EFMNUV_`���
���
���
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
"�
inputs/3���������
"�
inputs/4���������
"�
inputs/5���������
"�
inputs/6���������
p

 
� "�����������
$__inference_signature_wrapper_319981�%&-.56=>EFMNUV_`���
� 
���
*
aspect �
aspect���������
&
curv�
curv���������
0
	elevation#� 
	elevation���������
&
lito�
lito���������
(
slope�
slope���������
$
twi�
twi���������
.
uso_solo"�
uso_solo���������"1�.
,
dense_2!�
dense_2���������