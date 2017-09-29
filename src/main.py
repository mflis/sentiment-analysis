#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src import Config, Model, Placeholders, Runner

if __name__ == '__main__':
    config = Config()
    placeholders = Placeholders(config)
    model = Model(config, placeholders)

    runner = Runner.Runner(config, placeholders, model)
    runner.run()
