# Sentiment analysis


## Running through Ansible on AWS
 - install locally [Ansible](http://docs.ansible.com/ansible/latest/intro_installation.html#latest-releases-via-pip)
 - fill in `ansible/group_vars/all/vault-template.yml` and `ansible/export_aws-template.sh` with your AWS credentials
 -
 ```
 cp ansible/group_vars/all/vault-template.yml ansible/group_vars/all/vault.yml
 cp ansible/export_aws-template.sh ansible/export_aws.sh
 ```
 - adjust `regions ` in `ansible\ec2.ini` to name of region you use
 -
```
cd ansible
 ansible-playbook site.yml --extra-vars='{"params":"epochs=20", "type":"cnn"}'  --private-key=path-to-your-aws-key.pem
```
  - `type`  - type of network to run. Possible values: `cnn`, `tfidf`
  - `params` - parameters to change default configuration of network. Possible values to change are listed in `cnn-config.yaml` and `tfidf-config.yaml` for cnn and tfidf network respectively
  - Other configurable parameters of Ansible are listed in `ansible/roles/role-name/defaults/main.yml`. They can be set either through `--extra-vars` or in `ansible/group_vars/all/all.yml`