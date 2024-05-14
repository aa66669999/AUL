This is the classwork for deep learning. all the codes wrote by our group members. 
We creat many .py file and they have their own work.
let's focus on the main file:
1. get data:
get_data(train_ratio=args.train, pool_ratio=args.pool,test_ratio=args.test)
2. model:
   our backbone model: ResNet18(in_size=num_features, hidden_size=args.m_hidden, out_size=out_size, embed=args.m_embed,
                        drop_p=args.m_drop_p, activation=args.m_activation).to(device)
   
   baseline backbone model:BernoulliMixture(in_size=num_features, hidden_size=32, out_size=out_size,
                       embed_length=args.m_embed, drop_p=args.m_drop_p, activation=nn.ELU()).to(device)
   
   our AUC optimization loss function: criterion = ml_nn_loss2

   baseline backbone loss: criterion = ml_nn_loss
   
4. train model: 
model_opt, loss_opt = train(train_model,
                            dataloaders,
                            criterion=criterion,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            num_epochs=args.pretrain_epochs,
                            device_train=None,
                            num_l=num_labels,
                            fname=fnamesub
                            )
   
5. evaluate model:
   model_opt.eval()
   
6. visualize and document:
results += add_res(model_opt, test_data.get_x(), test_data.get_y(), device=device)
print(results)
with open('./result/' + fnamesub, 'a') as f:
    writer_obj = csv.writer(f)
    writer_obj.writerow(results)


