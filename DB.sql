create table if not exists user
(
    id        int auto_increment
        primary key,
    username  varchar(255)                 not null,
    password  varchar(255)                 not null,
    authority varchar(255) default 'admin' not null,
    constraint user_pk_2
        unique (username),
    constraint check_authority
        check (`authority` in ('admin','other'))
);

create table if not exists camera
(
    id     int auto_increment
        primary key,
    userId int          not null,
    name   varchar(255) not null,
    place  varchar(255) not null,
    type   varchar(255) not null,
    constraint camera_pk
        unique (name),
    constraint fk_user_id
        foreign key (userId) references user (id)
            on update cascade on delete cascade,
    constraint chk_valid_type
        check (`type` in ('face','danger'))
);

create table if not exists danger
(
    id       int auto_increment
        primary key,
    cameraId int not null,
    x1       int not null,
    x2       int not null,
    y1       int not null,
    y2       int not null,
    constraint fk_shape_camera
        foreign key (cameraId) references camera (id)
            on update cascade on delete cascade
);



create table if not exists face
(
    id     int auto_increment
        primary key,
    name   varchar(255) not null,
    x1     double       not null,
    x2     double       not null,
    x3     double       not null,
    x4     double       not null,
    x5     double       not null,
    x6     double       not null,
    x7     double       not null,
    x8     double       not null,
    x9     double       not null,
    x10    double       not null,
    x11    double       not null,
    x12    double       not null,
    x13    double       not null,
    x14    double       not null,
    x15    double       not null,
    x16    double       not null,
    x17    double       not null,
    x18    double       not null,
    x19    double       not null,
    x20    double       not null,
    x21    double       not null,
    x22    double       not null,
    x23    double       not null,
    x24    double       not null,
    x25    double       not null,
    x26    double       not null,
    x27    double       not null,
    x28    double       not null,
    x29    double       not null,
    x30    double       not null,
    x31    double       not null,
    x32    double       not null,
    x33    double       not null,
    x34    double       not null,
    x35    double       not null,
    x36    double       not null,
    x37    double       not null,
    x38    double       not null,
    x39    double       not null,
    x40    double       not null,
    x41    double       not null,
    x42    double       not null,
    x43    double       not null,
    x44    double       not null,
    x45    double       not null,
    x46    double       not null,
    x47    double       not null,
    x48    double       not null,
    x49    double       not null,
    x50    double       not null,
    x51    double       not null,
    x52    double       not null,
    x53    double       not null,
    x54    double       not null,
    x55    double       not null,
    x56    double       not null,
    x57    double       not null,
    x58    double       not null,
    x59    double       not null,
    x60    double       not null,
    x61    double       not null,
    x62    double       not null,
    x63    double       not null,
    x64    double       not null,
    x65    double       not null,
    x66    double       not null,
    x67    double       not null,
    x68    double       not null,
    x69    double       not null,
    x70    double       not null,
    x71    double       not null,
    x72    double       not null,
    x73    double       not null,
    x74    double       not null,
    x75    double       not null,
    x76    double       not null,
    x77    double       not null,
    x78    double       not null,
    x79    double       not null,
    x80    double       not null,
    x81    double       not null,
    x82    double       not null,
    x83    double       not null,
    x84    double       not null,
    x85    double       not null,
    x86    double       not null,
    x87    double       not null,
    x88    double       not null,
    x89    double       not null,
    x90    double       not null,
    x91    double       not null,
    x92    double       not null,
    x93    double       not null,
    x94    double       not null,
    x95    double       not null,
    x96    double       not null,
    x97    double       not null,
    x98    double       not null,
    x99    double       not null,
    x100   double       not null,
    x101   double       not null,
    x102   double       not null,
    x103   double       not null,
    x104   double       not null,
    x105   double       not null,
    x106   double       not null,
    x107   double       not null,
    x108   double       not null,
    x109   double       not null,
    x110   double       not null,
    x111   double       not null,
    x112   double       not null,
    x113   double       not null,
    x114   double       not null,
    x115   double       not null,
    x116   double       not null,
    x117   double       not null,
    x118   double       not null,
    x119   double       not null,
    x120   double       not null,
    x121   double       not null,
    x122   double       not null,
    x123   double       not null,
    x124   double       not null,
    x125   double       not null,
    x126   double       not null,
    x127   double       not null,
    x128   double       not null,
    userId int          not null,
    constraint face_user_id_fk
        foreign key (userId) references user (id)
            on update cascade on delete cascade
);

create index user_username_index
    on user (username);

create table if not exists warning
(
    id       int auto_increment
        primary key,
    cameraId int                                 not null,
    curTime  timestamp default CURRENT_TIMESTAMP null,
    videoURL varchar(255)                        not null,
    info     varchar(255)                        not null,
    type     varchar(255)                        not null,
    constraint fk_event_camera
        foreign key (cameraId) references camera (id),
    constraint chk_valid_event_type
        check (`type` in ('stranger','cheat','helmet','dangerous area','tumble'))
);

