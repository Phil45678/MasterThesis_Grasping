import gpflow
import trieste 
import tensorflow as tf 


class NatGradTrainedVGP(trieste.models.VariationalGaussianProcess):
    def optimize(self, dataset):
        gpflow.set_trainable(self.model.q_mu, False)
        gpflow.set_trainable(self.model.q_sqrt, False)
        variational_params = [(self.model.q_mu, self.model.q_sqrt)]
        adam_opt = tf.optimizers.Adam(1e-3)
        natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=0.1)

        for step in range(50):
            loss = self.model.training_loss
            natgrad_opt.minimize(loss, variational_params)
            adam_opt.minimize(loss, self.model.trainable_variables)


