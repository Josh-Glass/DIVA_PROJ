## external requirements
import numpy as np




def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))

def linear(x):
    return x

def relu(x):
    return x * (x > 0)

def get_formatted_targets(inputs, labels):
    targets = np.zeros([inputs.shape[0], inputs.shape[1]*2])   # np.full makes 8 by 6 array and fills it with target value
    for i in range(inputs.shape[0]):
        if labels[i] == 0:
            targets[i,:3] = inputs[i,:]
        elif labels[i] == 1:
            targets[i,3:] = inputs[i,:]
    return targets

data = {
    #'shj1': np.genfromtxt('data/shj/shj1.csv', delimiter = ','),
    'shj2': np.genfromtxt('data/shj/shj2.csv', delimiter = ','),
    #'shj3': np.genfromtxt('data/shj/shj3.csv', delimiter = ','),
    'shj4': np.genfromtxt('data/shj/shj4.csv', delimiter = ','),
    #'shj5': np.genfromtxt('data/shj/shj5.csv', delimiter = ','),
    #'shj6': np.genfromtxt('data/shj/shj6.csv', delimiter = ','),
}


behavioral_all_structures = np.genfromtxt('data/shj/revised1vs4.csv', delimiter = ',')

behavioral_all_structures = behavioral_all_structures[1:,:]








def get_fit(learn_rate, num_hidden_nodes, weight_range, beta):
    # # # # # # # # # # 
    # 
    # SETUP
    # 
    # # # # # # # # # #
    fit_err = 0
    num_epochs = 8
    inits= 100





    ## "forward pass"
    def forward(params, inputs, channel):
        hidden_act_raw = np.add(
            np.matmul(
                inputs,
                params['input']['hidden']['weights']
            ),
            params['input']['hidden']['bias']
        )

        hidden_act = sigmoid(hidden_act_raw)

        output_act_raw = np.add(
            np.matmul(
                hidden_act,
                params['hidden'][channel]['weights']
            ),
            params['hidden'][channel]['bias'],
        )

        output_act = sigmoid(output_act_raw)

        return [hidden_act_raw, hidden_act, output_act_raw, output_act]


    ## cost function (sum squared error)
    def loss(params, inputs, channel, targets = None):
        if np.any(targets) == None: targets = inputs
        return np.sum(
            np.square(
                np.subtract(
                    forward(params, inputs, channel)[-1],
                    targets
                )
            )
        ) / inputs.shape[0]




    ## backprop (for sum squared error cost function)
    def loss_grad(params, input, channel, targets = None):
        if np.any(targets) == None: targets = inputs
        inputs = np.reshape(input, (1,3))         

        hidden_act_raw, hidden_act, output_act_raw, output_act = forward(params, inputs, channel)

        ## gradients for decode layer ( chain rule on cost function )
        decode_grad = np.multiply(
            sigmoid_deriv(output_act_raw),
            (2 * (output_act - targets))  / inputs.shape[0] # <-- deriv of cost function
        )

        ## gradients for decode bias
        decode_grad_b = decode_grad.sum(axis = 0, keepdims = True)

        ## gradients for decode weights
        decode_grad_w = np.matmul(
            hidden_act.T,
            decode_grad
        )

        # - - - - - - - - - - - -

        ## gradients for encode layer ( chain rule on hidden layer )
        encode_grad = np.multiply(
            sigmoid_deriv(hidden_act_raw),
            np.matmul(
                decode_grad, 
                params['hidden'][channel]['weights'].T
            )
        )
        
        ## gradients for encode weights
        encode_grad_w = np.matmul(
            inputs.T,
            encode_grad
        )

        ## gradients for encode bias
        encode_grad_b = encode_grad.sum(axis = 0, keepdims = True)

        return {
            'input': {
                'hidden': {
                    'weights': encode_grad_w,
                    'bias': encode_grad_b,
                }
            },
            'hidden': {
                channel: {
                    'weights': decode_grad_w,
                    'bias': decode_grad_b,
                }
            }
        }



    ## luce choice w/ late-stage attention
    def response(params, inputs, channels, targets = None, beta = beta):
        if np.any(targets) == None: targets = inputs

        activations = np.array([
            forward(params, inputs, channel)[-1]
            for channel in channels
        ])

        # get beta weights using paired differences
        diversities = np.abs(
            np.diff(activations, axis = 0)
        ).sum(axis = 0)

        ## exponentiate & weight (beta) diversities
        diversities = np.exp(
            beta * diversities
        )

        ## softmax diversities
        fweights = diversities / np.sum(diversities)

        channel_errors = np.sum(
            np.square(
                np.subtract(
                    targets,
                    activations
                )
            ) * fweights,
            axis = 2, keepdims = True
        )

        return (1 / channel_errors) / np.sum(1 / channel_errors, axis = 0, keepdims = True)

    ## build parameter dictionary
    def build_params(num_features, num_hidden_nodes, categories, weight_range_low, weight_range_high ): # <-- he et al (2015) initialization
        '''
        num_features <-- (numeric) number of feature in the dataset
        num_hidden_nodes <-- (numeric)
        num_categories <-- number of category channels to make
        '''
        return {
            'input': {
                'hidden': {
                    'weights': np.random.uniform(weight_range_low, weight_range_high, [num_features, num_hidden_nodes]),
                    'bias': np.random.uniform(weight_range_low, weight_range_high,[1, num_hidden_nodes]),
                },
            },
            'hidden': {
                **{
                    channel: {
                        'weights': np.random.uniform(weight_range_low,weight_range_high, [num_hidden_nodes, num_features]),
                        'bias': np.random.uniform(weight_range_low,weight_range_high, [1, num_features]),
                    }
                    for channel in categories
                }
            },
        }
    ## weight update
    def update_params(params, gradients, lr):
        for layer in params:
            for connection in gradients[layer]:
                params[layer][connection]['weights'] -= lr * gradients[layer][connection]['weights']
                params[layer][connection]['bias'] -= lr * gradients[layer][connection]['bias']
        return params

    ## predict
    def predict(params, inputs, categories, targets = None):
        if np.any(targets) == None: targets = inputs
        return np.argmax(
            response(params, inputs, categories, targets),
            axis = 0
        )






    ####################################
    ####################################
    #########   RUN  THE MODEL #########
    ####################################
    ####################################








    # This is the outer most loop, it loops through all the different shj types
    #initalize fit errors list outside of the loop so that it DOES NOT reset after going through each shj type
    fit_errors = []
    for s, structure in enumerate(data):

        struct = data[structure]
        idx1 = np.arange(struct.shape[0])
        

        #Just shuffling the data
        np.random.shuffle(idx1)
        
        first8 = struct[idx1]
        
        struct = first8

        inputs = struct[:,:-1]
        

        labels = (struct[:,-1] - 1).astype(int)
            
        behavioral = behavioral_all_structures[0:8,[s]].reshape(8, 1)

        targets = inputs / 2 + .5 #not sure why we're dividing by this term here<-- ask about this

        categories = np.unique(labels)
    

        #make an array to hold the performance data
        #want the array to hold the data for one shj type at a time, retaining data across all inits
        #so initalize before the init loop, so it only resets when the shj type changes
        performance_data = np.zeros([num_epochs, inits])

        
        #create a presentation order
        presentation_order = np.arange(inputs.shape[0])

        

        
        #This second level loop loops through all of the inits for the current shj structure
        for init in range(inits):
            
            #this array will keep track of accuracy for each epoch within an init
            #it can reset after each init, so initalize within init but before epoch loop
            acc_array = np.zeros([num_epochs, 1])

            #build new params for each initialization
            params = build_params(
            num_features= inputs.shape[1],
            num_hidden_nodes=num_hidden_nodes, 
            categories=categories, 
            weight_range_low=-weight_range,
            weight_range_high=weight_range)
            





            #This third level loop loops through all of the epochs (training blocks)
            for e in range(num_epochs): 
                
                #shuffle the presentation order at the beginning of each epoch
                np.random.shuffle(presentation_order)
                
                #initialize accuracy list before presentation loop, so that it resets after each epoch
                acc_score_per_epoch =[]
                #This fourth level loop loops through each item in a epoch/block
                for p in presentation_order:
                    


                   ## Step 1: Record Model Response
                    pred = predict(params = params, inputs = inputs[p,:], categories = categories, targets = targets[p,:])
                    if pred == labels[p]:
                        acc_score_per_item = 1
                    else:
                        acc_score_per_item = 0

                    acc_score_per_epoch.append(acc_score_per_item)
            


                    ## Step 2: Update Model Weights
                    gradients = loss_grad(params, inputs[p,:], labels[p], targets = targets[p,:])
                    params = update_params(params, gradients, learn_rate)
                #add the average resp prob of correct items for the current epoch 
                acc_array[e] = np.mean(acc_score_per_epoch)
                #store epoch by init resp prob data 
                performance_data[e,init] = acc_array[e]
        
        #take the average across all inits per epoch<--left with array of size 8X1
        accuracy = performance_data.mean(axis = 1).reshape(8, 1)
        

        
        fit_errors.append(np.sum( (accuracy - behavioral) ** 2 ))
        fit_err = np.sum(fit_errors)

        
        
    return fit_err





#print(get_fit(learn_rate=2.0, num_hidden_nodes=8, weight_range=2.37, beta=450))

