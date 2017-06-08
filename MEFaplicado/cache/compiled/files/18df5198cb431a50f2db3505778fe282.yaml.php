<?php
return [
    '@class' => 'Grav\\Common\\File\\CompiledYamlFile',
    'filename' => '/home/markinho/Dropbox/argenta-web.github.io/MEFaplicado/user/config/system.yaml',
    'modified' => 1496927900,
    'data' => [
        'home' => [
            'alias' => '/inicio'
        ],
        'languages' => [
            'http_accept_language' => true,
            'supported' => [
                0 => 'en',
                1 => 'pt-BR'
            ]
        ],
        'pages' => [
            'theme' => 'learn2',
            'markdown_extra' => true,
            'process' => [
                'markdown' => true,
                'twig' => false
            ]
        ],
        'cache' => [
            'enabled' => true,
            'check' => [
                'method' => 'file'
            ],
            'driver' => 'auto',
            'prefix' => 'g'
        ],
        'twig' => [
            'cache' => true,
            'debug' => false,
            'auto_reload' => true,
            'autoescape' => false
        ],
        'assets' => [
            'css_pipeline' => true,
            'css_minify' => true,
            'css_rewrite' => true,
            'js_pipeline' => true,
            'js_minify' => true
        ],
        'debugger' => [
            'enabled' => true,
            'twig' => true,
            'shutdown' => [
                'close_connection' => true
            ]
        ]
    ]
];
