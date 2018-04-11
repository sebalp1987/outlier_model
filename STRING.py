
"""
STRING FILE. It contains paths and relevant Strings used in the project.
"""


'--------------PATH FILES ---------------------------------------------------------------------'

path_processing = 'doc_output\\file_preprocessed'
path_final_files = 'file_preprocessed\\batch_files\\'
path_test = 'doc_output\\file_preprocessed\\test_files'


'--------------DUMMIES FILLNA -----------------------------------------------------------------------'
fillna_id = ['d_producto_', 'd_entidad_', 'd_tipodoc_']
fillna_cliente_hogar = []
fillna_cliente = ['cliente_forma_contacto_', 'cliente_telefono_tipo_', 'cliente_region_', 'cliente_residencia_region_']
fillna_fecha = ['d_fecha_siniestro_ocurrencia_year', 'dfecha_siniestro_ocurrencia_month_',
                'd_fecha_siniestro_ocurrencia_weekday_', 'd_fecha_poliza_efecto_natural_year_',
                'd_fecha_poliza_vto_natural_year_']
fillna_hist_mov_pol_otro = ['d_hist_poliz_ultimo_movimiento_', 'd_hist_poliza_estado_A_',
                            'd_hist_mov_poliza_otro_producto_ACCIDENTES_']
fillna_hist_mov_pol_ref = ['d_hist_movimiento_tipo_', 'd_hist_movimiento_tipo_']
fillna_hist_ant_otro = ['d_hist_sin_poliza_otro_producto_']
fillna_hist_sin_ant_ref = ['d_hist_sin_sit_']
fillna_hist_sin_ref = ['hist_siniestro_actual_oficina_']
fillna_hogar = ['d_tipo_hogar_', 'd_hogar_ubicacion_', 'd_hogar_caracter_', 'd_hogar_uso_']
fillna_pago = ['d_pago_canal_cobro_1er_recibo_', 'd_pago_situacion_recibo_', 'd_pago_morosidad_', 'd_pago_forma_curso_',
               ]
fillna_perito = []
fillna_reserva = []
fillna_poliza = ['poliza_desc_estructura', 'poliza_canal', 'poliza_duracion', 'poliza_credit_scoring',
                 'poliza_ultimo_movimiento']
fillna_siniestro = []
fillna_europa = []

fillna_vars = fillna_id + fillna_cliente_hogar + fillna_cliente + fillna_fecha + fillna_hist_mov_pol_otro + \
              fillna_hist_mov_pol_ref + fillna_hist_ant_otro + fillna_hist_sin_ant_ref + fillna_hist_sin_ref + \
              fillna_hogar + fillna_pago + fillna_perito + fillna_reserva + fillna_poliza + fillna_siniestro


'--------------HEADERS -----------------------------------------------------------------------'

header_id = ['ID_SINIESTRO', 'ID_FISCAL', 'ID_POLIZA', 'ID_PRODUCTO', 'ID_DOSSIER']


header_blacklist = ['ID_SINIESTRO', 'ID_POLIZA', 'ID_PRODUCTO', 'FECHA_APERTURA',
                    'FECHA_TERMINADO', 'NIF_TOMADOR', 'TOMADOR', 'NIF_PAGADOR', 'PAGADOR',
                    'CUENTA_IBAN_PRINCIPAL', 'COD_INTERMEDIARIO', 'ROL', 'DESC_ROL', 'NIF'
                    ]

header_blacklist_tomador = ['ID_SINIESTRO', 'ID_POLIZA', 'ID_PRODUCTO', 'FECHA_APERTURA',
                            'FECHA_TERMINADO', 'NIF', 'NOMBRE', 'NIF_PAG', 'NOMBRE_PAG']

header_blacklist_iban = ['ID_SINIESTRO', 'ID_POLIZA', 'IBAN']

header_blacklist_intermediario = ['ID_SINIESTRO', 'ID_POLIZA', 'COD_INTERMEDIARIO']

header_blacklist_rol = ['ID_SINIESTRO', 'ID_POLIZA', 'ROL', 'DESCR_ROL','NIF']


'--------------PARAMETROS -----------------------------------------------------------------------'


class Parameters:
    import time
    end_date = time.strftime('%Y-%m-%d')
    init_date = '2015-01-01'
    init_date_new = '2017-01-01'
    sinister_year = 2014
    correct_city_names = {'CALPE' : 'CALP', 'ISIL':'els-pallaresos', 'MARTORELLES':'MARTORELL', 'SANTA GERTRUDIS':'IBIZA',
                          'CIUDADELA' : 'PALMA', 'MORELL, EL':'TARRAGONA', 'ENCINAREJO DE CORDOB':'CORDOBA',
                          'PORTALS NOUS':'PALMA', 'OÃATI':'OÑATE', 'VILADEMAT':'GIRONA', 'VILADAMAT':'GIRONA',
                          'BORRASA':'BORRASSA', 'EREÑO':'BILBAO', 'SANTA MARGARIDA':'PALMA', 'BAQUEIRA':'BENASQUE',
                          'SALITJA':'GIRONA', 'VILADECNAS': 'VILADECANS', 'QUEIXANS':'ALP',
                          'CALETA DE FUSTE':'LAS PALMAS DE GRAN CANARIA', 'CELEIRO':'BURELA DE CABO',
                          'QUINTES':'GIJON', 'CASTELLON':'valencia-valencia', 'CALPE': 'alicante-valencia',
                          'GILENA': 'malaga', 'ESPINOSA DE VILLAGONZALO S/N': 'burgos',
                          'SANTA MARGARIDA DE MONTBUI': 'sabadell',
                          "L'ALCUDIA": 'valencia-valencia', "ALCUDIA, L'": 'valencia-valencia'}

    cp = ['cliente_cp', 'hogar_cp']

    fecha = ['fecha_poliza_emision', 'fecha_poliza_efecto_natural', 'fecha_poliza_efecto_mvto',
             'fecha_poliza_vto_movimiento', 'fecha_poliza_vto_natural', 'fecha_siniestro_ocurrencia',
             'fecha_primera_visita_peritaje', 'fecha_ultima_visita_peritaje',
             'hist_siniestro_poliza_otro_ultimo_fecha_ocurrencia', 'hist_siniestro_otro_ultimo_fecha_ocurrencia',
             'fecha_primer_peritaje', 'fecha_ultimo_informe']

