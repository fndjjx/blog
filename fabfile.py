#!/usr/bin/env python
# encoding: utf-8

from fabric.api import local,cd,run,env

env.hosts=['bloger@106.14.24.66:22',]



def update_remote():
    print("remote update")
    with cd('~/git_repo/blog'): 
        run('git pull --rebase')
        run('supervisorctl -c supervisord.conf shutdown')
        run('supervisord -c supervisord.conf')

def update():
    update_remote()
