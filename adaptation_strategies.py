import torch


class AdaptationStrategies:
    none = 'none'
    tent = 'tent'
    cotta = 'cotta'
    cpt4 = 'cpt4'

    @classmethod
    def adapt(cls, x, model, original_model, model_ema=None, optimizer=None,
              adaptation_strategy=None, prompt_key_loss_coef=1, clip_grad=1.0, mt_alpha=0.99):

        output_adaptation = None

        if adaptation_strategy == cls.none:
            output_adaptation = cls.no_adaptation(x, model, original_model)
        elif adaptation_strategy == cls.tent:
            output_adaptation = cls.tent_adaptation(x, model, original_model, optimizer, prompt_key_loss_coef)
        elif adaptation_strategy == cls.cotta:
            output_adaptation = cls.cotta_adaptation(x, model, original_model, model_ema, optimizer,
                                                     prompt_key_loss_coef, clip_grad, mt_alpha)

        return output_adaptation

    @classmethod
    def no_adaptation(cls, x, model, original_model):
        model.eval()
        with torch.no_grad():
            if original_model is not None:
                output = original_model(x)
                cls_features = output['pre_logits']
            else:
                cls_features = None

        output = model(x, cls_features=cls_features)
        return output['logits']

    @classmethod
    def tent_adaptation(cls, x, model, original_model, optimizer, prompt_key_loss_coef):
        with torch.no_grad():
            if original_model is not None:
                output = original_model(x)
                cls_features = output['pre_logits']
            else:
                cls_features = None

        model.train()
        output = model(x, cls_features=cls_features)
        loss = (-(output['logits'].softmax(1) * output['logits'].log_softmax(1)).sum(1)).mean(0)
        if 'reduce_sim' in output and prompt_key_loss_coef:
            loss = loss + prompt_key_loss_coef * output['reduce_sim']

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return output['logits']

    @classmethod
    def cotta_adaptation(cls, x, model, original_model, model_ema, optimizer,
                         prompt_key_loss_coef, clip_grad, mt_alpha):
        with torch.no_grad():
            if original_model is not None:
                output = original_model(x)
                cls_features = output['pre_logits']
            else:
                cls_features = None

        outputs = model(x, cls_features=cls_features)
        outputs_ema = model_ema(x, cls_features=cls_features)

        # Teacher Prediction
        # standard_ema = self.model_ema(x, cls_features=cls_features)

        # Augmentation-averaged Prediction
        # TODO Augmentation
        # anchor_prob = torch.nn.functional.softmax(self.model_anchor(x)['logits'], dim=1).max(1)[0]
        # if anchor_prob.mean(0) < self.ap:
        #     N = 32
        #     outputs_emas = []
        #     for i in range(N):
        #         outputs_ = self.model_ema(self.transform(x))['logits'].detach()
        #         outputs_emas.append(outputs_)
        #
        #     outputs_ema = torch.stack(outputs_emas).mean(0)
        # else:
        # outputs_ema = standard_ema

        # Student Update
        loss = (cls.softmax_entropy(outputs['logits'], outputs_ema['logits'])).mean(0)
        if 'reduce_sim' in outputs_ema and prompt_key_loss_coef:
            loss = loss + prompt_key_loss_coef * outputs_ema['reduce_sim']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        torch.nn.utils.clip_grad_norm_(model_ema.parameters(), clip_grad)
        optimizer.step()

        # Teacher Update
        model_ema = cls.update_ema_variables(ema_model=model_ema, model=model, alpha_teacher=mt_alpha)

        # for k, ms in cls.model_state.items():
        #     if ms.device != 'cuda':
        #         cls.model_state[k] = ms.cuda()
        #
        # # Stochastic Restore
        # for nm, m in model.named_modules():
        #     for npp, p in m.named_parameters():
        #         if npp in ['weight', 'bias', 'prompt', 'prompt_key', 'batch_norm'] and p.requires_grad:
        #             mask = (torch.rand(p.shape) < mt_alpha).float().cuda()
        #             with torch.no_grad():
        #                 p.data = cls.model_state[f"{nm}.{npp}"] * mask + p * (1. - mask)
        #                 pass
        return outputs_ema['logits']

    @classmethod
    def softmax_entropy(cls, x, x_ema):
        """Entropy of softmax distribution from logits."""
        return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

    @classmethod
    def update_ema_variables(cls, ema_model, model, alpha_teacher):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
        return ema_model
