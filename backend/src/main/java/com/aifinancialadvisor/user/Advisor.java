package com.aifinancialadvisor.user;

import java.util.UUID;

public class Advisor {

    private UUID id;
    private String name;
    private String email;

    public Advisor(String name, String email) {
        this.id = UUID.randomUUID();
        this.name = name;
        this.email = email;
    }

    public UUID getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public String getEmail() {
        return email;
    }
}