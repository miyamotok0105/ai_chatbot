drop table if exists conversation_pair;
create table conversation_pair (
    id integer primary key autoincrement not null,
    user1 text not null,
    user1_talk text not null,
    user2 text not null,
    user2_talk text not null
);

