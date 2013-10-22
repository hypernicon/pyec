"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

try:
    import theano
except ImportError:
    print "Could not find Theano. Falling back on CPU-based RNN instance. If you want",
    print "to use a GPU, please `easy_install Theano`"
    raise
else:    
    from datetime import datetime
    
    import theano.tensor as T
    
    import numpy as np
    
    TIDENTITY = 0
    TLOGISTIC = 1
    TTHRESHOLD = 2
    TBIAS = 3
    TRADIAL = 4
    THYPERBOLIC = 5
    
    topologyCache = {}
    
    def RnnEvaluator(weights, topo):
        """Build a Theano function that computes the internal state of the network
        when called.
        
        """
        #start = datetime.now()
        
        numLayers = len(weights)
        numInputs = 0
        numOutputs = 0
        ws = []
        weightArgs = []
        parts = []
        for neurons, activator, isInput, isOutput, weightFrame in weights:
            if isInput:
                numInputs += 1
            if isOutput:
                numOutputs += 1
            srcs = []
            for i, info in enumerate(weightFrame):
                ws.append(T.fmatrix())
                srcIdx, w = info
                weightArgs.append(w)
            parts.append(len(ws))
        
        if topo in topologyCache:
            net = topologyCache[topo]
        else:
            #print "CACHE MISS! ", topo
            
            def recast_weights(ws):
                expanded = []
                last = 0
                for part in parts:
                    expanded.append(ws[last:part])
                    last = part
                return expanded
        
            def evaluate_net(*args):
                states = args[:numLayers]
                pweights = recast_weights(args[numLayers:])
                activations = T.fvectors(numLayers)
                idx = 0
                for neurons, activator, isInput, isOutput, weightFrame in weights:
                    sumParts = []
                    layerWeights = pweights[idx]
                    for i, info in enumerate(weightFrame):
                        srcIdx, _ = info
                        w = layerWeights[i]
                        sumParts.append(T.dot(states[srcIdx], w.transpose()))
                
                    if len(sumParts):
                        sumParts = T.stack(*sumParts)
                        activity = T.sum(sumParts, axis=0)
                        if activator == TIDENTITY:
                            activation = activity
                        elif activator == TLOGISTIC:
                            activation = 1. / (1. + T.exp(-activity))
                        elif activator == THYPERBOLIC:
                            activation = T.tanh(activity)
                        elif activator == TTHRESHOLD:
                            activation = T.sgn(activity)
                        elif activator == TBIAS:
                            activation = T.ones_like(activity, dtype='float32')
                        elif activator == TRADIAL:
                            activation = T.exp(-activity*activity/2.0)
                        else:
                            raise Exception("Unknown activation function for layer {0}" + layer.id)
                    else:
                        activation = T.zeros_like(states[idx])#states[idx]
                    
                    activations[idx] = activation
                    idx += 1
            
                checklist = [T.all(T.eq(a,s)) for a,s in zip(activations, states)]
                condition = T.all(T.as_tensor_variable(checklist))
                return activations, {}, theano.scan_module.until(condition )
        
            def make_states(*inputs):
                states = []
                idx = 0
                numPoints = len(inputs) and inputs[0].shape[0] or 1
                for neurons, activator, isInput, isOutput, weightFrame in weights:
                    if isInput:
                        states.append(inputs[idx])
                        idx += 1
                    else:
                        states.append(T.ones((numPoints,neurons), dtype='float32'))
                return states
        
            def project_output(states):
                outputs = []
                idx = 0
                for neurons, activator, isInput, isOutput, weightFrame in weights:
                    if isOutput:
                        outputs.append(states[idx])
                    idx += 1
                return outputs
        
            inputs = T.fmatrices(numInputs)
            times = T.iscalar()
            netValue, updates = theano.scan(
                fn=evaluate_net,
                outputs_info=make_states(*inputs),
                non_sequences=ws,
                n_steps=times
            )
        
            result = [n[-1] for n in netValue]
        
            outputs = project_output(result)
        
            #profmode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker()) 
            net = theano.function(inputs + ws + [times], outputs)#, mode=profmode)
            
            topologyCache[topo] = net
 
        def fix_inputs(inputs, times=5):
            reshape = False
            if len(inputs) and (len(np.shape(inputs[0])) == 1):
                reshape = True
                inputs = [np.reshape(i, (1,i.shape[0])) for i in inputs]
            args = list(inputs) + weightArgs + [times]
            outputs = net(*args)
            #profmode.print_summary()
            if reshape:
                return [o[0] for o in outputs]
            return outputs
        
        #print "compilation took: ", (datetime.now() - start).total_seconds()
        return fix_inputs
